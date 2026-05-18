"""
model_registry.py — Stage −1: Model Registry
═════════════════════════════════════════════
Implements Stage −1 of algorithm_sm_inference.md.

Quickstart
----------
    from model_registry import ModelRegistry

    registry = ModelRegistry(
        noise_json     = 'noise_models.json',
        diffusion_json = 'diffusion_models.json',
        model_dir      = '/path/to/model/files',
    )
    registry.load_all()

    noise_model = registry.noise_models.best_for(has_raw=True)
    d_model     = registry.diffusion_models.best_for(
                      trace_length_ms=60_000, has_raw=True)

Model catalogue files
---------------------
Two JSON files define all available models:

  noise_models.json — list of noise model entries:
    {"label", "window_ms", "mape", "requires_raw", "dt_channels",
     "feat_dim", "requires_acf_features",
     "model_file", "scaler_mean", "scaler_scale"}

  diffusion_models.json — list of diffusion model entries (same schema +
    optional "conformal_intervals" block for uncertainty intervals):
    {"conformal_intervals": {"coverage": 0.90,
       "pad_strata": [{"pad_max": 0.25, "q_log10": ...}, ...]}}

ModelWrapper interface (common to noise and diffusion wrappers)
---------------------------------------------------------------
    .metadata             — dict from JSON
    .label                — str
    .window_ms            — int
    .mape                 — float
    .dt_channels          — list[float]
    .requires_raw         — bool
    .is_admissible(trace_length_ms, has_raw) → (bool, reason)
    .predict(win_min, win_050, win_100, features) → float
    .compute_features(trace) → np.ndarray  (if requires_acf_features)
    .conformal_intervals  — ConformaIntervals or None (diffusion only)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ── Conformal interval lookup ─────────────────────────────────────────────────

@dataclass
class ConformalIntervals:
    """
    Pre-computed conformal prediction quantiles per pad_fraction stratum.
    Populated from the 'conformal_intervals' field in diffusion_models.json.
    """
    coverage:   float         # nominal coverage, e.g. 0.90
    pad_maxes:  List[float]   # upper bound of each stratum, e.g. [0.25, 0.50, 0.75, 1.0]
    q_log10:    List[float]   # 90th-pctile of |log10(D_pred) - log10(D_true)| per stratum

    def lookup(self, pad_fraction: float) -> float:
        """
        Return the conformal quantile for the given pad_fraction.
        Uses the lowest stratum whose pad_max >= pad_fraction.
        """
        for pmax, q in zip(self.pad_maxes, self.q_log10):
            if pad_fraction <= pmax:
                return q
        return self.q_log10[-1]   # fallback: highest pad stratum

    @classmethod
    def from_dict(cls, d: dict) -> 'ConformalIntervals':
        strata   = d['pad_strata']
        pad_maxes = [s['pad_max']  for s in strata]
        q_log10  = [s['q_log10'] for s in strata]
        return cls(coverage=d.get('coverage', 0.90),
                   pad_maxes=pad_maxes, q_log10=q_log10)


# ── Model wrapper ─────────────────────────────────────────────────────────────

class ModelWrapper:
    """
    Wraps a single PyTorch model with a standardised inference interface.

    Parameters
    ----------
    meta      : dict — JSON entry from the model catalogue
    model_dir : Path — directory containing .pt and .npy scaler files
    kind      : 'noise' or 'diffusion'
    """

    def __init__(self, meta: dict, model_dir: Path, kind: str = 'diffusion'):
        self.metadata   = meta
        self._model_dir = Path(model_dir)
        self.kind       = kind

        # Core metadata fields
        self.label                 = meta['label']
        self.window_ms             = int(meta['window_ms'])
        self.mape                  = float(meta.get('mape', np.nan))
        self.dt_channels           = [float(x) for x in meta.get('dt_channels', [1.0])]
        self.requires_raw          = bool(meta.get('requires_raw', False))
        self.feat_dim              = meta.get('feat_dim', 0)
        self.requires_acf_features = bool(meta.get('requires_acf_features', False))

        # Pytorch model (loaded lazily by load_all)
        self._net        = None
        self._device     = None
        self._loaded     = False
        self._load_error = None

        # Feature scalers
        self._scaler_mean  = None
        self._scaler_scale = None

        # Channel affinity (None = any channel; 'S1' or 'S2' = channel-specific)
        self.channel_affinity: Optional[str] = meta.get('channel_affinity', None)

        # Conformal intervals (diffusion models only)
        self.conformal_intervals: Optional[ConformalIntervals] = None
        if 'conformal_intervals' in meta:
            try:
                self.conformal_intervals = ConformalIntervals.from_dict(
                    meta['conformal_intervals']
                )
            except (KeyError, TypeError) as e:
                log.warning(f'{self.label}: bad conformal_intervals in JSON: {e}')

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self, device=None) -> None:
        """
        Load model weights and scalers from disk.

        The .pt files are state_dict saves (not TorchScript).  We instantiate
        the correct nn.Module class via model_classes.build_model(), then call
        load_state_dict().  The JSON entry must specify:
          "model_class" : e.g. "WaveNetMultiDT_B"
          "init_kwargs" : dict of constructor keyword args (feat_dim, etc.)
        """
        import torch
        from model_classes import build_model

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._device = device

        model_file = self._model_dir / self.metadata.get('model_file', '')
        if not model_file.exists():
            self._load_error = f'model file not found: {model_file}'
            log.warning(f'{self.label}: {self._load_error}')
            return

        class_name = self.metadata.get('model_class')
        init_kwargs = self.metadata.get('init_kwargs', {})

        if class_name is None:
            self._load_error = 'missing "model_class" in JSON entry'
            log.warning(f'{self.label}: {self._load_error}')
            return

        try:
            net = build_model(class_name, **init_kwargs)
            state = torch.load(str(model_file), map_location=device,
                               weights_only=True)
            net.load_state_dict(state)
            net.to(device)
            net.eval()
            self._net = net
        except Exception as e:
            self._load_error = str(e)
            log.warning(f'{self.label}: failed to load — {e}')
            return

        # Load scalers (feature normalisation)
        for attr, key in [('_scaler_mean',  'scaler_mean'),
                          ('_scaler_scale', 'scaler_scale')]:
            fname = self.metadata.get(key)
            if fname:
                fp = self._model_dir / fname
                if fp.exists():
                    setattr(self, attr, np.load(fp))
                else:
                    log.warning(f'{self.label}: scaler file not found: {fp}')

        # Load raw-stat scalers (noise models use different key names)
        for attr, key in [('_scaler_mean',  'raw_stat_mean'),
                          ('_scaler_scale', 'raw_stat_std')]:
            fname = self.metadata.get(key)
            if fname and getattr(self, attr) is None:
                fp = self._model_dir / fname
                if fp.exists():
                    setattr(self, attr, np.load(fp))

        self._loaded = True
        log.info(f'{self.label}: loaded on {device}')

    # ── Admissibility ─────────────────────────────────────────────────────────

    def is_admissible(
        self,
        trace_length_ms: float = np.inf,
        has_raw: bool = True,
    ) -> Tuple[bool, str]:
        """
        Check whether this model can be applied to the given data.

        Returns
        -------
        (admissible: bool, reason: str)
        """
        if not self._loaded:
            return False, 'not loaded'
        if self._load_error:
            return False, self._load_error
        if trace_length_ms < self.window_ms:
            return False, (f'trace too short: {trace_length_ms:.0f}ms < '
                           f'{self.window_ms}ms required')
        if self.requires_raw and not has_raw:
            return False, 'requires RAW files (has_raw=False)'
        return True, ''

    # ── Feature computation ───────────────────────────────────────────────────

    def compute_features(self, trace: np.ndarray) -> np.ndarray:
        """
        Compute the ACF/scalar feature vector for a single trace window.

        The exact lag set, normalisation, and feature ordering are defined
        by the model; Stage 4 calls this method without knowing the details.

        Parameters
        ----------
        trace : np.ndarray, shape (N,) — intensity window at the finest dt

        Returns
        -------
        features : np.ndarray, shape (feat_dim,)
        """
        if not self.requires_acf_features:
            # Simple 6-feature model: [mean_min, std_min, mean_050, std_050,
            #                          mean_100, std_100]
            # (the model uses 6 raw stats; pass trace as-is and compute here)
            mu  = float(trace.mean())
            sig = float(trace.std())
            return np.array([mu, sig, mu, sig, mu, sig], dtype=np.float64)

        # ACF-based features — delegate to compute_features.py for the
        # standard 302-feature pipeline used by the multidt_B models.
        # The feature pipeline is model-specific; for now import the shared
        # implementation and call it.
        try:
            from compute_acf_multidt_features import compute_features as _cf
        except ImportError:
            # Fallback to the single-dt pipeline (212 features → 302 with extras)
            from compute_features import compute_features as _cf

        feats = _cf(trace)
        return np.asarray(feats, dtype=np.float64)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        traces:   dict,
        features: Optional[np.ndarray] = None,
    ) -> float:
        """
        Run inference on one model input window.

        Parameters
        ----------
        traces   : dict mapping dt_ms (float) → np.ndarray (1-D intensity window)
                   e.g. {0.1: arr_dt010, 0.5: arr_dt050, 1.0: arr_dt100}
                   Keys must match self.dt_channels.
        features : np.ndarray of shape (feat_dim,), un-normalised.
                   Scaler normalisation applied internally.

        Returns
        -------
        float
            Diffusion models → D in µm²/s   (wrapper exponentiates ln output)
            Noise models     → n_abs in kHz  (Softplus × N_SCALE / 1000)

        Raises
        ------
        RuntimeError if the model is not loaded.
        ValueError   if a required dt channel is missing from traces.
        """
        import torch

        if not self._loaded or self._net is None:
            raise RuntimeError(f'{self.label}: model not loaded')

        def _norm(arr: np.ndarray) -> torch.Tensor:
            """Per-trace z-score normalisation matching training augmentation."""
            a   = arr.astype(np.float32)
            mu  = a.mean()
            sig = a.std() + 1e-8
            return torch.from_numpy((a - mu) / sig).unsqueeze(0).to(self._device)

        # Build ordered trace tensors (ascending dt)
        trace_args = []
        for dt in sorted(self.dt_channels):
            arr = traces.get(dt)
            if arr is None:
                raise ValueError(
                    f'{self.label}: missing trace for dt={dt}ms '
                    f'(provided keys: {sorted(traces)})'
                )
            trace_args.append(_norm(arr))

        # Compute or accept feature vector
        if features is None and self.requires_acf_features:
            # Compute features internally.  Traces are in kHz (counts/ms); the
            # sim training data was stored as counts/s (counts/dt_s, dt_s=1e-3),
            # so multiply by 1000 to match the numerical range the scaler was
            # fitted on.  The WaveNet receives the original kHz values above
            # (z-scored via _norm, so scale-invariant).
            from compute_features_multidt import compute_features_for_trace
            features = np.concatenate([
                compute_features_for_trace(traces[dt] * 1000.0)
                for dt in sorted(self.dt_channels)
            ])
        if features is not None:
            f = features.astype(np.float64)
            if self._scaler_mean is not None:
                f = (f - self._scaler_mean) / (self._scaler_scale + 1e-9)
            feats_t = torch.from_numpy(f.astype(np.float32)).unsqueeze(0).to(self._device)
        else:
            feats_t = None

        # Forward pass: positional args = (*trace_tensors, feats_tensor)
        with torch.no_grad():
            args = trace_args + ([feats_t] if feats_t is not None else [])
            out  = self._net(*args)

        raw = float(out.squeeze().cpu())

        if self.kind == 'noise':
            # Model trained with target = n_abs_hz / N_SCALE (N_SCALE=1e3).
            # raw_output × N_SCALE / 1000 = raw_output → already in kHz.
            return raw          # kHz
        else:
            return float(np.exp(raw))   # D in µm²/s  (model outputs ln D)

    def predict_batch(
        self,
        traces_list:   list,
        features_list: Optional[list] = None,
        batch_size:    int = 128,
    ) -> list:
        """
        Batch inference on multiple windows.

        Parameters
        ----------
        traces_list   : list of dicts mapping dt_ms (float) → np.ndarray
        features_list : list of feature arrays (one per window), or None
        batch_size    : mini-batch size for GPU memory management

        Returns
        -------
        list of floats — D in µm²/s (diffusion) or n_abs in kHz (noise)
        """
        import torch

        if not self._loaded or self._net is None:
            raise RuntimeError(f'{self.label}: model not loaded')

        n = len(traces_list)
        if n == 0:
            return []

        results = []
        for start in range(0, n, batch_size):
            end  = min(start + batch_size, n)
            b_tr = traces_list[start:end]
            b_ft = (features_list[start:end]
                    if features_list is not None else None)

            # Build one stacked tensor per dt channel: shape (B, L_dt)
            trace_args = []
            for dt in sorted(self.dt_channels):
                arrs = []
                for t in b_tr:
                    arr = t[dt].astype(np.float32)
                    mu  = arr.mean()
                    sig = arr.std() + 1e-8
                    arrs.append((arr - mu) / sig)
                trace_args.append(
                    torch.from_numpy(np.stack(arrs)).to(self._device)
                )

            # Feature tensor: shape (B, feat_dim)
            feats_t = None
            if self.requires_acf_features:
                from compute_features_multidt import compute_features_for_trace
                if b_ft is None or b_ft[0] is None:
                    # Compute internally with kHz → counts/s scaling
                    b_ft = [
                        np.concatenate([
                            compute_features_for_trace(t[dt] * 1000.0)
                            for dt in sorted(self.dt_channels)
                        ])
                        for t in b_tr
                    ]
                fa = np.stack([f.astype(np.float64) for f in b_ft])
                if self._scaler_mean is not None:
                    fa = (fa - self._scaler_mean) / (self._scaler_scale + 1e-9)
                feats_t = torch.from_numpy(fa.astype(np.float32)).to(self._device)

            with torch.no_grad():
                args = trace_args + ([feats_t] if feats_t is not None else [])
                out  = self._net(*args)

            raws = np.atleast_1d(out.squeeze(-1).cpu().numpy())
            if self.kind == 'noise':
                results.extend(raws.tolist())
            else:
                results.extend(float(np.exp(r)) for r in raws)

        return results

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self):
        status = 'loaded' if self._loaded else ('error' if self._load_error else 'not loaded')
        return (f'ModelWrapper({self.label!r}, kind={self.kind}, '
                f'W={self.window_ms}ms, MAPE={self.mape:.1f}%, {status})')


# ── Model collection helpers ──────────────────────────────────────────────────

class ModelCollection:
    """
    A dict-like collection of ModelWrapper objects with a best_for() selector.

    Parameters
    ----------
    wrappers : list[ModelWrapper]
    """

    def __init__(self, wrappers: list):
        self._wrappers: Dict[str, ModelWrapper] = {w.label: w for w in wrappers}

    def __getitem__(self, label: str) -> ModelWrapper:
        return self._wrappers[label]

    def __iter__(self):
        return iter(self._wrappers.values())

    def values(self):
        return self._wrappers.values()

    def items(self):
        return self._wrappers.items()

    def best_for(
        self,
        trace_length_ms: float = np.inf,
        has_raw: bool = True,
        channel_id: Optional[str] = None,
    ) -> Optional[ModelWrapper]:
        """
        Return the admissible model with the lowest MAPE.

        Parameters
        ----------
        trace_length_ms : duration of the trace being processed
        has_raw         : whether RAW photon-arrival files are available
        channel_id      : detector channel label (e.g. 'S1', 'S2').
                          Models whose channel_affinity does not match are
                          excluded.  Models with channel_affinity=None are
                          considered channel-agnostic and always included.
        """
        candidates = []
        for w in self._wrappers.values():
            if not w.is_admissible(trace_length_ms, has_raw)[0]:
                continue
            if channel_id is not None and w.channel_affinity is not None:
                if w.channel_affinity != channel_id:
                    continue
            candidates.append(w)
        if not candidates:
            return None
        return min(candidates, key=lambda w: w.mape)


# ── Registry ──────────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Loads and manages all noise and diffusion models for the inference pipeline.

    Attributes
    ----------
    noise_models      : ModelCollection
    diffusion_models  : ModelCollection
    dt_min            : float — finest dt channel across all models (ms)
    """

    def __init__(
        self,
        noise_json:     str | Path,
        diffusion_json: str | Path,
        model_dir:      str | Path,
    ):
        self._noise_json     = Path(noise_json)
        self._diffusion_json = Path(diffusion_json)
        self._model_dir      = Path(model_dir)

        self.noise_models:     Optional[ModelCollection] = None
        self.diffusion_models: Optional[ModelCollection] = None
        self.dt_min:           float                     = 1.0   # updated on load

    def load_all(self, device=None) -> None:
        """
        Parse catalogue JSONs, instantiate wrappers, and load all model weights.
        Prints a summary table of loaded / failed models.
        """
        noise_list     = self._parse_json(self._noise_json)
        diffusion_list = self._parse_json(self._diffusion_json)

        noise_wrappers = [
            ModelWrapper(m, self._model_dir, kind='noise')
            for m in noise_list
        ]
        diff_wrappers = [
            ModelWrapper(m, self._model_dir, kind='diffusion')
            for m in diffusion_list
        ]

        all_wrappers = noise_wrappers + diff_wrappers
        for w in all_wrappers:
            w.load(device=device)

        self.noise_models     = ModelCollection(noise_wrappers)
        self.diffusion_models = ModelCollection(diff_wrappers)

        # Compute dt_min from all loaded models
        all_dts = [
            dt
            for w in all_wrappers
            if w._loaded
            for dt in w.dt_channels
        ]
        self.dt_min = min(all_dts) if all_dts else 1.0

        self._print_summary(all_wrappers)

    def _parse_json(self, path: Path) -> list:
        if not path.exists():
            log.warning(f'Catalogue not found: {path}')
            return []
        with open(path) as f:
            return json.load(f)

    def _print_summary(self, wrappers: list) -> None:
        print(f'\n{"Model":<28} {"Kind":<12} {"W(ms)":<8} {"MAPE%":<8} {"Status"}')
        print('-' * 70)
        for w in sorted(wrappers, key=lambda x: (x.kind, x.mape)):
            status = 'OK' if w._loaded else f'FAIL: {w._load_error}'
            print(f'{w.label:<28} {w.kind:<12} {w.window_ms:<8} '
                  f'{w.mape:<8.1f} {status}')
        print(f'\ndt_min = {self.dt_min} ms')

    def __repr__(self):
        n = sum(1 for w in (list(self.noise_models.values())
                             + list(self.diffusion_models.values()))
                if w._loaded) if self.noise_models else 0
        return (f'ModelRegistry({n} models loaded, dt_min={self.dt_min}ms)')
