#!/usr/bin/env python3
"""
gui.py — SM Pipeline Manager
Two-tab GUI:
  • Bookkeeper  — survey experiments, view processing history
  • Orchestrator — configure, submit, monitor and retrieve pipeline runs
"""

import json
import queue
import shutil
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bookkeeper  import survey, REGISTRY_FILENAME
from orchestrator import (ssh, rsync_file_up, rsync_files_up, rsync_up,
                          rsync_outputs_down, rsync_new_outputs_down,
                          list_remote_processed_folders,
                          active_job_ids, job_final_state,
                          select_experiments)
import importlib
import version as _version_mod
from version import PIPELINE_VERSION, VERSION_REGISTRY, CLUSTER_MANIFEST

def _reload_version():
    """Reload version.py from disk and return (VERSION_REGISTRY, CLUSTER_MANIFEST,
    PIPELINE_VERSION) so callers always see the current file without restarting."""
    importlib.reload(_version_mod)
    return (_version_mod.VERSION_REGISTRY,
            _version_mod.CLUSTER_MANIFEST,
            _version_mod.PIPELINE_VERSION)

DEFAULT_CLUSTER     = 'bouchet.ycrc.yale.edu'
DEFAULT_REMOTE_BASE = '/nfs/roberts/project/pi_sah46/ah2286/new_'
SLURM_SCRIPT        = 'job_full_pipeline_with_plots.sh'
SESSION_PREFIX      = 'pipeline_session_'   # timestamped; multiple may coexist
SCRIPT_DEFAULT_TIME = '02:00:00'            # matches job_full_pipeline_with_plots.sh


def _bump_time(current, extra_hours=2):
    """Add extra_hours to a SLURM HH:MM:SS time string.

    If current is empty (script default is in effect), SCRIPT_DEFAULT_TIME
    is used as the base before bumping.
    """
    base   = current.strip() if current and current.strip() else SCRIPT_DEFAULT_TIME
    parts  = base.split(':')
    h      = int(parts[0]) if len(parts) >= 1 else 0
    m      = int(parts[1]) if len(parts) >= 2 else 0
    s      = int(parts[2]) if len(parts) >= 3 else 0
    total  = h * 3600 + m * 60 + s + extra_hours * 3600
    nh, nm = divmod(total, 3600)
    nm, ns = divmod(nm, 60)
    return f'{nh:02d}:{nm:02d}:{ns:02d}'


def _fmt_elapsed(secs):
    """Format an integer number of seconds as H:MM:SS (or M:SS when < 1 h)."""
    h, rem = divmod(int(secs), 3600)
    m, s   = divmod(rem, 60)
    if h > 0:
        return f'{h}:{m:02d}:{s:02d}'
    return f'{m}:{s:02d}'


def _parse_version(ver_str):
    """Parse a pipeline version string into a numeric tuple for comparison.

    Handles leading 'v', multi-part minor versions, and non-integer suffixes.
    Examples:
        'v3.5'  → (3, 5)
        'v4.31' → (4, 31)
        '3.9'   → (3, 9)
        ''      → (0, 0)
    """
    s = (ver_str or '').strip().lstrip('v')
    parts = s.split('.')
    try:
        return tuple(int(p) for p in parts if p.isdigit() or p.lstrip('-').isdigit())
    except (ValueError, AttributeError):
        return (0, 0)


# ── Helpers ───────────────────────────────────────────────────────────────────

class _LogQueue:
    """Thread-safe string queue for passing log lines to the GUI."""
    def __init__(self):
        self._q = queue.Queue()

    def put(self, msg, tag='normal'):
        self._q.put((msg, tag))

    def drain(self):
        items = []
        while True:
            try:
                items.append(self._q.get_nowait())
            except queue.Empty:
                break
        return items


# ── Bookkeeper tab ────────────────────────────────────────────────────────────

class BookkeeperTab(ttk.Frame):
    def __init__(self, parent, shared_dir_var):
        super().__init__(parent)
        self._dir = shared_dir_var
        self._build()

    def _build(self):
        # ── Top bar ───────────────────────────────────────────────────────────
        bar = ttk.Frame(self)
        bar.pack(fill='x', padx=10, pady=(10, 4))

        ttk.Label(bar, text='Parent directory:').pack(side='left')
        ttk.Entry(bar, textvariable=self._dir, width=54).pack(side='left', padx=4)
        ttk.Button(bar, text='Browse…', command=self._browse).pack(side='left')
        ttk.Button(bar, text='Survey', command=self._survey,
                   style='Accent.TButton').pack(side='left', padx=(8, 0))

        # ── Treeview ──────────────────────────────────────────────────────────
        frm = ttk.Frame(self)
        frm.pack(fill='both', expand=True, padx=10, pady=4)

        cols = ('experiment', 'run_folder', 'version', 'status', 'files', 'events')
        self._tree = ttk.Treeview(frm, columns=cols, show='headings', selectmode='browse')

        widths = dict(experiment=340, run_folder=210, version=65,
                      status=130, files=55, events=65)
        for col in cols:
            label = col.replace('_', ' ').title()
            anchor = 'w' if col in ('experiment', 'run_folder') else 'center'
            self._tree.heading(col, text=label, anchor=anchor)
            self._tree.column(col, width=widths[col], anchor=anchor, stretch=(col == 'experiment'))

        vsb = ttk.Scrollbar(frm, orient='vertical', command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')

        # Row colour tags
        self._tree.tag_configure('completed',     foreground='#2E7D32')
        self._tree.tag_configure('failed',        foreground='#C62828')
        self._tree.tag_configure('not_processed', foreground='#757575')
        self._tree.tag_configure('unknown',       foreground='#E65100')
        self._tree.tag_configure('bold',          font=('TkDefaultFont', 10, 'bold'))

        # ── Cleanup panel ─────────────────────────────────────────────────────
        clup = ttk.LabelFrame(self, text='Cleanup old runs  (local only — Bouchet untouched)')
        clup.pack(fill='x', padx=10, pady=(2, 4))
        cr = ttk.Frame(clup)
        cr.pack(fill='x', padx=6, pady=4)
        ttk.Label(cr, text='Delete runs before version:').pack(side='left')
        self._cleanup_ver_var = tk.StringVar(value='')
        ttk.Entry(cr, textvariable=self._cleanup_ver_var, width=10).pack(side='left', padx=4)
        ttk.Button(cr, text='Preview', command=self._cleanup_preview).pack(side='left', padx=(0, 4))
        ttk.Button(cr, text='Delete…', command=self._cleanup_delete).pack(side='left')
        self._cleanup_info = tk.StringVar(value='')
        ttk.Label(cr, textvariable=self._cleanup_info,
                  foreground='#E65100').pack(side='left', padx=10)

        # ── Status bar ────────────────────────────────────────────────────────
        self._status = tk.StringVar(value='Ready — enter a parent directory and click Survey.')
        ttk.Label(self, textvariable=self._status, relief='sunken',
                  anchor='w').pack(fill='x', padx=10, pady=(0, 6))

    # ── Actions ───────────────────────────────────────────────────────────────

    def _browse(self):
        d = filedialog.askdirectory(title='Select parent experiments directory')
        if d:
            self._dir.set(d)

    def _survey(self):
        p = Path(self._dir.get().strip())
        if not p.is_dir():
            messagebox.showerror('Not found', f'Directory not found:\n{p}')
            return
        try:
            reg = survey(p)
        except Exception as exc:
            messagebox.showerror('Survey error', str(exc))
            return

        # Persist registry
        with open(p / REGISTRY_FILENAME, 'w') as f:
            json.dump(reg, f, indent=2)

        self._populate(reg)

    def _populate(self, registry):
        self._tree.delete(*self._tree.get_children())
        experiments = registry.get('experiments', {})

        for exp_name in sorted(experiments):
            runs = experiments[exp_name]['runs']
            if not runs:
                self._tree.insert('', 'end',
                    values=(exp_name, '—', '—', 'not processed', '—', '—'),
                    tags=('not_processed', 'bold'))
            else:
                for i, run in enumerate(runs):
                    status = run.get('status') or 'unknown'
                    tag    = status.replace(' ', '_')
                    self._tree.insert('', 'end', values=(
                        exp_name if i == 0 else '',
                        run.get('run_folder', '—'),
                        run.get('pipeline_version') or '—',
                        status,
                        run['n_measurements'] if run.get('n_measurements') is not None else '—',
                        run['n_events']       if run.get('n_events')       is not None else '—',
                    ), tags=(tag, 'bold') if i == 0 else (tag,))

        n   = len(experiments)
        ts  = registry.get('last_surveyed', '')
        self._status.set(f'{n} experiment(s) found  |  last surveyed: {ts}')

    # ── Cleanup helpers ───────────────────────────────────────────────────────

    def _get_cleanup_targets(self):
        """Return (cutoff_tuple, list_of_(exp_dir, run_dir)) or (None, None) on error."""
        cutoff_str = self._cleanup_ver_var.get().strip()
        if not cutoff_str:
            messagebox.showerror('Missing version',
                                 'Enter a cutoff version (e.g. v3.5) first.')
            return None, None

        cutoff = _parse_version(cutoff_str)
        if cutoff == (0, 0):
            messagebox.showerror('Invalid version',
                                 f'Cannot parse version: "{cutoff_str}"\n'
                                 'Use a format like  v3.5  or  4.1')
            return None, None

        parent_dir = Path(self._dir.get().strip())
        if not parent_dir.is_dir():
            messagebox.showerror('Not found',
                                 f'Survey a directory first:\n{parent_dir}')
            return None, None

        targets = []
        for exp_dir in sorted(p for p in parent_dir.iterdir()
                               if p.is_dir() and not p.name.startswith('.')):
            for run_dir in sorted(p for p in exp_dir.iterdir()
                                  if p.is_dir() and p.name.startswith('processed_on_')):
                log_path = run_dir / 'pipeline_log.json'
                if not log_path.exists():
                    continue   # no log — skip (safe)
                try:
                    with open(log_path) as f:
                        log = json.load(f)
                    ver_str = log.get('pipeline_version') or ''
                    run_ver = _parse_version(ver_str)
                    if run_ver and run_ver < cutoff:
                        targets.append((exp_dir, run_dir, ver_str or '?'))
                except Exception:
                    pass   # corrupted log — skip

        return cutoff, targets

    def _cleanup_preview(self):
        cutoff, targets = self._get_cleanup_targets()
        if cutoff is None:
            return
        cutoff_str = self._cleanup_ver_var.get().strip()
        n = len(targets)
        if n == 0:
            self._cleanup_info.set(f'Nothing to delete before {cutoff_str}.')
            return

        # Count disk usage per target
        lines = [f'{n} folder(s) would be deleted (version < {cutoff_str}):\n']
        for exp_dir, run_dir, ver in targets:
            lines.append(f'  [{ver}]  {exp_dir.name} / {run_dir.name}')

        dlg = tk.Toplevel(self)
        dlg.title('Cleanup preview — local only')
        dlg.resizable(True, True)
        txt = scrolledtext.ScrolledText(dlg, width=90, height=20,
                                        font=('Courier', 10), state='normal')
        txt.pack(fill='both', expand=True, padx=8, pady=8)
        txt.insert('end', '\n'.join(lines))
        txt.configure(state='disabled')
        ttk.Button(dlg, text='Close', command=dlg.destroy).pack(pady=(0, 8))
        self._cleanup_info.set(f'{n} folder(s) found before {cutoff_str}.')

    def _cleanup_delete(self):
        cutoff, targets = self._get_cleanup_targets()
        if cutoff is None:
            return
        cutoff_str = self._cleanup_ver_var.get().strip()
        n = len(targets)
        if n == 0:
            messagebox.showinfo('Nothing to delete',
                                f'No local run folders found with version < {cutoff_str}.')
            self._cleanup_info.set('Nothing to delete.')
            return

        preview_lines = '\n'.join(
            f'  [{ver}]  {exp_dir.name} / {run_dir.name}'
            for exp_dir, run_dir, ver in targets)
        confirmed = messagebox.askyesno(
            'Confirm local deletion',
            f'Permanently delete {n} run folder(s) with version < {cutoff_str}?\n\n'
            f'{preview_lines}\n\n'
            '⚠  This only affects your local machine.\n'
            '   All copies on Bouchet remain untouched.\n\n'
            'Continue?',
            icon='warning')
        if not confirmed:
            self._cleanup_info.set('Cancelled.')
            return

        deleted, errors = 0, []
        for exp_dir, run_dir, ver in targets:
            try:
                shutil.rmtree(run_dir)
                deleted += 1
            except Exception as exc:
                errors.append(f'{run_dir.name}: {exc}')

        if errors:
            messagebox.showerror('Errors during deletion',
                                 f'Deleted {deleted}/{n}.  Errors:\n' +
                                 '\n'.join(errors))
        self._cleanup_info.set(f'Deleted {deleted} folder(s).')

        # Re-survey to refresh the treeview
        self._survey()

    def refresh(self, parent_dir: Path):
        """Called externally (e.g. after orchestrator finishes) to re-survey."""
        self._dir.set(str(parent_dir))
        self._survey()


# ── Orchestrator tab ──────────────────────────────────────────────────────────

class OrchestratorTab(ttk.Frame):
    def __init__(self, parent, shared_dir_var, bookkeeper_tab):
        super().__init__(parent)
        self._dir              = shared_dir_var
        self._bk_tab           = bookkeeper_tab
        self._submit_cancel    = threading.Event()   # cancelled per submission only
        self._submitting       = False               # True only during phases 1-4
        self._logq             = _LogQueue()
        self._job_status_times = {}   # exp_name → (status, time.monotonic())
        self._build()
        self._poll()
        self._tick_elapsed()
        self.after(0, self._check_session)  # enable Resume/Delete if sessions exist on startup

    def _build(self):
        # ── Settings ──────────────────────────────────────────────────────────
        sf = ttk.LabelFrame(self, text='Settings')
        sf.pack(fill='x', padx=10, pady=(10, 4))

        def row(label, var, width=50, widget='entry', **kw):
            r = ttk.Frame(sf)
            r.pack(fill='x', padx=6, pady=2)
            ttk.Label(r, text=label, width=20, anchor='e').pack(side='left')
            if widget == 'entry':
                ttk.Entry(r, textvariable=var, width=width).pack(side='left', padx=4)
            elif widget == 'spin':
                ttk.Spinbox(r, textvariable=var, width=width, **kw).pack(side='left', padx=4)
            return r

        self._ver_var        = tk.StringVar(value=PIPELINE_VERSION)
        self._all_var        = tk.BooleanVar(value=False)
        self._cluster_var    = tk.StringVar(value=DEFAULT_CLUSTER)
        self._remote_var     = tk.StringVar(value=DEFAULT_REMOTE_BASE)
        self._remote_exp_var = tk.StringVar(value='')
        self._poll_var       = tk.IntVar(value=60)
        self._time_var       = tk.StringVar(value='')

        # Parent dir row (shared)
        r0 = ttk.Frame(sf); r0.pack(fill='x', padx=6, pady=2)
        ttk.Label(r0, text='Parent directory:', width=20, anchor='e').pack(side='left')
        ttk.Entry(r0, textvariable=self._dir, width=50).pack(side='left', padx=4)
        ttk.Button(r0, text='Browse…', command=self._browse).pack(side='left')

        r1 = row('Target version:', self._ver_var, width=12)
        self._all_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(r1, text='Reprocess all  (--all)',
                        variable=self._all_var).pack(side='left', padx=12)

        row('Cluster:', self._cluster_var, width=40)
        row('Remote base:', self._remote_var, width=50)

        r4 = row('Remote experiments:', self._remote_exp_var, width=50)
        ttk.Label(r4, text='(blank → <remote-base>/experiments)',
                  foreground='gray').pack(side='left', padx=4)

        row('Poll interval (s):', self._poll_var, width=6, widget='spin',
            from_=10, to=600)

        r5 = row('Time limit (override):', self._time_var, width=12)
        ttk.Label(r5, text='(blank = script default 2:00:00, e.g. 4:00:00)',
                  foreground='gray').pack(side='left', padx=4)

        # Watch parent dir changes to detect resumable sessions
        self._dir.trace_add('write', lambda *_: self.after(100, self._check_session))

        # ── Buttons ───────────────────────────────────────────────────────────
        bf = ttk.Frame(self)
        bf.pack(fill='x', padx=10, pady=4)
        self._run_btn    = ttk.Button(bf, text='▶  Run Orchestrator', command=self._run)
        self._cancel_btn = ttk.Button(bf, text='■  Cancel', command=self._do_cancel,
                                      state='disabled')
        self._resume_btn = ttk.Button(bf, text='↩  Resume Session', command=self._resume,
                                      state='disabled')
        self._delete_sess_btn = ttk.Button(bf, text='🗑  Delete Session…',
                                           command=self._delete_session_ui,
                                           state='disabled')
        self._run_btn.pack(side='left')
        self._cancel_btn.pack(side='left', padx=6)
        self._resume_btn.pack(side='left', padx=6)
        self._delete_sess_btn.pack(side='left', padx=6)
        self._prog_var = tk.StringVar(value='')
        ttk.Label(bf, textvariable=self._prog_var,
                  foreground='#1565C0').pack(side='left', padx=10)

        # ── Progress indicator ────────────────────────────────────────────────
        pf = ttk.Frame(self)
        pf.pack(fill='x', padx=10, pady=(0, 2))
        self._phase_var = tk.StringVar(value='')
        ttk.Label(pf, textvariable=self._phase_var,
                  foreground='#1565C0').pack(side='left')
        self._pbar = ttk.Progressbar(self, orient='horizontal',
                                     mode='determinate', length=100)
        self._pbar.pack(fill='x', padx=10, pady=(0, 6))

        # ── Job status table ──────────────────────────────────────────────────
        jf = ttk.LabelFrame(self, text='Job status')
        jf.pack(fill='x', padx=10, pady=4)

        jcols = ('experiment', 'job_id', 'version', 'status', 'elapsed')
        self._jtree = ttk.Treeview(jf, columns=jcols, show='headings', height=5)
        self._jtree.heading('experiment', text='Experiment', anchor='w')
        self._jtree.heading('job_id',     text='Job ID',     anchor='center')
        self._jtree.heading('version',    text='Version',    anchor='center')
        self._jtree.heading('status',     text='Status',     anchor='center')
        self._jtree.heading('elapsed',    text='Elapsed',    anchor='center')
        self._jtree.column('experiment', width=310, anchor='w',      stretch=True)
        self._jtree.column('job_id',     width=90,  anchor='center', stretch=False)
        self._jtree.column('version',    width=80,  anchor='center', stretch=False)
        self._jtree.column('status',     width=120, anchor='center', stretch=False)
        self._jtree.column('elapsed',    width=75,  anchor='center', stretch=False)
        self._jtree.tag_configure('queued',    foreground='#E65100')
        self._jtree.tag_configure('running',   foreground='#1565C0')
        self._jtree.tag_configure('completed', foreground='#2E7D32')
        self._jtree.tag_configure('failed',    foreground='#C62828')
        self._jtree.tag_configure('timeout',   foreground='#FF6F00')
        self._jtree.tag_configure('resubmitted', foreground='#6A1B9A')
        self._jtree.pack(fill='x', padx=4, pady=4)

        # ── Log ───────────────────────────────────────────────────────────────
        lf = ttk.LabelFrame(self, text='Log')
        lf.pack(fill='both', expand=True, padx=10, pady=(4, 10))
        self._log_text = scrolledtext.ScrolledText(
            lf, state='disabled', font=('Courier', 10), wrap='word')
        self._log_text.pack(fill='both', expand=True, padx=4, pady=4)
        self._log_text.tag_configure('phase',  foreground='#1565C0',
                                     font=('Courier', 10, 'bold'))
        self._log_text.tag_configure('ok',     foreground='#2E7D32')
        self._log_text.tag_configure('error',  foreground='#C62828')
        self._log_text.tag_configure('normal', foreground='black')

    # ── GUI thread helpers ────────────────────────────────────────────────────

    def _browse(self):
        d = filedialog.askdirectory(title='Select parent experiments directory')
        if d:
            self._dir.set(d)

    def _poll(self):
        """Drain log queue every 100 ms on the main thread."""
        for msg, tag in self._logq.drain():
            self._log_text.configure(state='normal')
            self._log_text.insert('end', msg + '\n', tag)
            self._log_text.see('end')
            self._log_text.configure(state='disabled')
        self.after(100, self._poll)

    def _tick_elapsed(self):
        """Update the Elapsed column for every job row once per second."""
        now = time.monotonic()
        for row in self._jtree.get_children():
            exp_name = self._jtree.set(row, 'experiment')
            if exp_name in self._job_status_times:
                _, t0 = self._job_status_times[exp_name]
                self._jtree.set(row, 'elapsed', _fmt_elapsed(now - t0))
        self.after(1000, self._tick_elapsed)

    def _clear_job_rows(self, exp_names=None):
        """Remove rows from the job table.  If exp_names is None, clear all rows."""
        def _do():
            to_remove = []
            for row in self._jtree.get_children():
                if exp_names is None or self._jtree.set(row, 'experiment') in exp_names:
                    to_remove.append(row)
            for row in to_remove:
                exp = self._jtree.set(row, 'experiment')
                self._jtree.delete(row)
                self._job_status_times.pop(exp, None)
        self.after(0, _do)

    def _set_job_row(self, exp_name, job_id, status, version=None):
        def _do():
            now = time.monotonic()
            for row in self._jtree.get_children():
                if self._jtree.set(row, 'experiment') == exp_name:
                    old_status = self._jtree.set(row, 'status')
                    self._jtree.set(row, 'job_id', job_id)
                    self._jtree.set(row, 'status', status)
                    if version is not None:
                        self._jtree.set(row, 'version', version)
                    self._jtree.item(row, tags=(status,))
                    if old_status != status:
                        self._job_status_times[exp_name] = (status, now)
                        self._jtree.set(row, 'elapsed', '0:00')
                    return
            # New row — start elapsed clock from now
            self._job_status_times[exp_name] = (status, now)
            self._jtree.insert('', 'end',
                values=(exp_name, job_id, version or PIPELINE_VERSION, status, '0:00'),
                tags=(status,))
        self.after(0, _do)

    def _set_submitting(self, active: bool):
        """Disable Run/Resume during phases 1-4; re-enable when monitoring begins."""
        def _do():
            self._submitting = active
            self._run_btn.configure(   state='disabled' if active else 'normal')
            self._cancel_btn.configure(state='normal'   if active else 'disabled')
            # Resume stays disabled during submission; re-check sessions otherwise
            if not active:
                self._check_session()
        self.after(0, _do)

    def _set_progress(self, msg: str):
        self.after(0, lambda: self._prog_var.set(msg))

    def _log(self, msg: str, tag: str = 'normal'):
        self._logq.put(msg, tag)

    # ── Run / cancel ──────────────────────────────────────────────────────────

    def _start_phase(self, label: str):
        """Switch bar to indeterminate (bouncing) and update label."""
        def _do():
            self._phase_var.set(label)
            self._pbar.configure(mode='indeterminate')
            self._pbar.start(12)
        self.after(0, _do)

    def _phase_progress(self, label: str, value: int, maximum: int):
        """Switch bar to determinate and set value (used during Phase 5)."""
        def _do():
            self._pbar.stop()
            self._pbar.configure(mode='determinate', maximum=maximum, value=value)
            self._phase_var.set(label)
        self.after(0, _do)

    def _finish_progress(self, label: str = 'Done'):
        """Fill bar to 100 % and show final label."""
        def _do():
            self._pbar.stop()
            self._pbar.configure(mode='determinate', maximum=100, value=100)
            self._phase_var.set(label)
        self.after(0, _do)

    def _reset_progress(self):
        def _do():
            self._pbar.stop()
            self._pbar.configure(mode='determinate', maximum=100, value=0)
            self._phase_var.set('')
        self.after(0, _do)

    # ── Session persistence ───────────────────────────────────────────────────

    def _find_sessions(self, parent_dir=None):
        """Return sorted list of session Paths in parent_dir."""
        if parent_dir is None:
            p = self._dir.get().strip()
            if not p:
                return []
            parent_dir = Path(p)
        if not parent_dir.is_dir():
            return []
        return sorted(parent_dir.glob(f'{SESSION_PREFIX}*.json'))

    def _save_session(self, parent_dir, job_map, cluster,
                      remote_base, remote_exp_dir, poll_interval,
                      n_total, pipeline_version,
                      time_limit='', time_limit_map=None, pipeline_script='',
                      pre_folders_map=None):
        """Write a timestamped session file; returns the Path."""
        ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = parent_dir / f'{SESSION_PREFIX}{ts}.json'
        # Serialise pre_folders_map: set → sorted list for JSON
        pfm_serial = {exp: sorted(folders)
                      for exp, folders in (pre_folders_map or {}).items()}
        data = {
            'job_map':          job_map,
            'cluster':          cluster,
            'remote_base':      remote_base,
            'remote_exp_dir':   remote_exp_dir,
            'poll_interval':    poll_interval,
            'n_total':          n_total,
            'n_done':           0,
            'pipeline_version': pipeline_version,
            'pipeline_script':  pipeline_script,
            'time_limit':       time_limit,
            'time_limit_map':   time_limit_map or {},
            'pre_folders_map':  pfm_serial,
            'started_at':       datetime.now().isoformat(timespec='seconds'),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        return path

    def _update_session_done(self, session_path, n_done):
        if not session_path.exists():
            return
        with open(session_path) as f:
            data = json.load(f)
        data['n_done'] = n_done
        with open(session_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _update_session_jobs(self, session_path, job_map, time_limit_map):
        """Persist updated job_map and time_limit_map (called after resubmission)."""
        if not session_path.exists():
            return
        with open(session_path) as f:
            data = json.load(f)
        data['job_map']        = job_map
        data['time_limit_map'] = time_limit_map
        with open(session_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _submit_one_job(self, cluster, remote_base, remote_exp,
                        pipeline_script, time_limit):
        """Submit a single SLURM job; returns job_id string or None on failure."""
        time_opt = f'--time={time_limit} ' if time_limit else ''
        cmd = (f'cd {remote_base} && sbatch {time_opt}'
               f'--export=DATA_FOLDER={remote_exp},'
               f'PIPELINE_SCRIPT={pipeline_script} {SLURM_SCRIPT}')
        out, rc = ssh(cluster, cmd, check=False)
        if rc != 0 or 'Submitted' not in out:
            return None
        return out.strip().split()[-1]

    def _delete_session(self, session_path):
        if session_path.exists():
            session_path.unlink()
        self.after(0, self._check_session)

    def _check_session(self):
        """Enable Resume/Delete buttons if session files exist and not currently submitting."""
        if self._submitting:
            return
        sessions = self._find_sessions()
        has = len(sessions) > 0
        state = 'normal' if has else 'disabled'
        self._resume_btn.configure(state=state)
        self._delete_sess_btn.configure(state=state)
        if has and not self._prog_var.get():
            n = len(sessions)
            self._prog_var.set(
                f'↩  {n} resumable session(s) found — click Resume to reconnect')
        elif not has and self._prog_var.get().startswith('↩'):
            self._prog_var.set('')

    def _delete_session_ui(self):
        """Let the user select one or more session files to delete."""
        parent_dir = Path(self._dir.get().strip())
        sessions   = self._find_sessions(parent_dir)
        if not sessions:
            messagebox.showwarning('No sessions', 'No session files found.')
            return

        if len(sessions) == 1:
            sp = sessions[0]
            with open(sp) as f:
                d = json.load(f)
            if messagebox.askyesno(
                    'Delete session',
                    f'Delete session file?\n\n'
                    f'  {sp.name}\n'
                    f'  version {d.get("pipeline_version","—")}  '
                    f'started {d.get("started_at","—")}\n'
                    f'  {d.get("n_done",0)} / {d.get("n_total","?")} jobs copied back\n\n'
                    'The jobs on the cluster are not affected.',
                    icon='warning'):
                exps = list(d.get('job_map', {}).values())
                sp.unlink()
                self._clear_job_rows(exps)
                self._check_session()
            return

        # Multiple sessions — show picker with multi-select
        win = tk.Toplevel(self)
        win.title('Delete session files')
        win.geometry('660x300')
        win.resizable(True, True)
        win.grab_set()

        ttk.Label(win, text='Select session(s) to delete  (Ctrl-click for multiple):',
                  font=('TkDefaultFont', 11)).pack(pady=(12, 4), padx=12, anchor='w')

        cols = ('file', 'version', 'started', 'jobs', 'done')
        tree = ttk.Treeview(win, columns=cols, show='headings',
                            height=7, selectmode='extended')
        tree.heading('file',    text='Session file',  anchor='w')
        tree.heading('version', text='Version',       anchor='center')
        tree.heading('started', text='Started at',    anchor='center')
        tree.heading('jobs',    text='Jobs',          anchor='center')
        tree.heading('done',    text='Copied back',   anchor='center')
        tree.column('file',    width=230, anchor='w')
        tree.column('version', width=70,  anchor='center')
        tree.column('started', width=160, anchor='center')
        tree.column('jobs',    width=50,  anchor='center')
        tree.column('done',    width=80,  anchor='center')
        tree.pack(fill='both', expand=True, padx=12, pady=4)

        path_map = {}
        for sp in sessions:
            try:
                with open(sp) as f:
                    d = json.load(f)
            except Exception:
                d = {}
            iid = tree.insert('', 'end', values=(
                sp.name,
                d.get('pipeline_version', '—'),
                d.get('started_at', '—'),
                d.get('n_total', '—'),
                f'{d.get("n_done", 0)} / {d.get("n_total", "?")}',
            ))
            path_map[iid] = sp

        bf = ttk.Frame(win)
        bf.pack(pady=6)

        def _do_delete():
            sel = tree.selection()
            if not sel:
                messagebox.showwarning('Nothing selected',
                                       'Select at least one session to delete.',
                                       parent=win)
                return
            names = '\n'.join(f'  {path_map[iid].name}' for iid in sel)
            if not messagebox.askyesno(
                    'Confirm delete',
                    f'Delete {len(sel)} session file(s)?\n\n{names}\n\n'
                    'The jobs on the cluster are not affected.',
                    icon='warning', parent=win):
                return
            all_exps = []
            for iid in sel:
                try:
                    with open(path_map[iid]) as f:
                        d = json.load(f)
                    all_exps.extend(d.get('job_map', {}).values())
                    path_map[iid].unlink()
                except Exception:
                    pass
            win.destroy()
            self._clear_job_rows(all_exps)
            self._check_session()

        ttk.Button(bf, text='Delete selected', command=_do_delete).pack(side='left', padx=6)
        ttk.Button(bf, text='Cancel', command=win.destroy).pack(side='left')

    def _resume(self):
        parent_dir = Path(self._dir.get().strip())
        sessions   = self._find_sessions(parent_dir)
        if not sessions:
            messagebox.showwarning('No session', 'No session files found.')
            return
        if len(sessions) == 1:
            self._start_resume(parent_dir, sessions[0])
        else:
            self._pick_session(parent_dir, sessions)

    def _pick_session(self, parent_dir, sessions):
        """Show a picker dialog when multiple sessions exist."""
        win = tk.Toplevel(self)
        win.title('Select session to resume')
        win.geometry('620x260')
        win.resizable(False, False)
        win.grab_set()

        ttk.Label(win, text='Choose a session to resume:',
                  font=('TkDefaultFont', 11)).pack(pady=(12, 4), padx=12, anchor='w')

        cols = ('file', 'version', 'started', 'jobs', 'done')
        tree = ttk.Treeview(win, columns=cols, show='headings', height=6)
        tree.heading('file',    text='Session file',  anchor='w')
        tree.heading('version', text='Version',       anchor='center')
        tree.heading('started', text='Started at',    anchor='center')
        tree.heading('jobs',    text='Jobs',          anchor='center')
        tree.heading('done',    text='Done',          anchor='center')
        tree.column('file',    width=220, anchor='w')
        tree.column('version', width=70,  anchor='center')
        tree.column('started', width=160, anchor='center')
        tree.column('jobs',    width=50,  anchor='center')
        tree.column('done',    width=50,  anchor='center')
        tree.pack(fill='both', expand=True, padx=12, pady=4)

        path_map = {}
        for sp in sessions:
            with open(sp) as f:
                d = json.load(f)
            iid = tree.insert('', 'end', values=(
                sp.name,
                d.get('pipeline_version', '—'),
                d.get('started_at', '—'),
                d.get('n_total', '—'),
                d.get('n_done',  '—'),
            ))
            path_map[iid] = sp

        bf = ttk.Frame(win)
        bf.pack(pady=6)

        def _ok():
            sel = tree.selection()
            if not sel:
                messagebox.showwarning('Select a session', 'Please select a session first.',
                                       parent=win)
                return
            sp = path_map[sel[0]]
            win.destroy()
            self._start_resume(parent_dir, sp)

        ttk.Button(bf, text='Resume selected', command=_ok).pack(side='left', padx=6)
        ttk.Button(bf, text='Cancel', command=win.destroy).pack(side='left')
        tree.bind('<Double-1>', lambda _: _ok())

    def _start_resume(self, parent_dir, session_path):
        with open(session_path) as f:
            sess = json.load(f)

        job_map         = sess['job_map']
        cluster         = sess['cluster']
        remote_base     = sess['remote_base']
        remote_exp_dir  = sess['remote_exp_dir']
        poll_interval   = sess['poll_interval']
        n_total         = sess['n_total']
        n_done_saved    = sess.get('n_done', 0)
        sess_version    = sess.get('pipeline_version', '—')
        time_limit      = sess.get('time_limit', '')
        time_limit_map  = sess.get('time_limit_map', {})
        pipeline_script = sess.get('pipeline_script', '')
        # pre_folders_map absent in old sessions → None signals legacy fallback
        pfm_raw = sess.get('pre_folders_map', None)
        pre_folders_map = ({exp: set(folders) for exp, folders in pfm_raw.items()}
                           if pfm_raw is not None else None)

        if not messagebox.askyesno(
                'Resume session',
                f'Reconnect to {n_total} job(s)  (version {sess_version})\n'
                f'submitted at {sess.get("started_at", "unknown time")}?\n\n'
                f'{n_done_saved} of {n_total} already copied back.'):
            return

        self._reset_progress()
        self._set_submitting(True)   # disable Run during catch-up phase
        threading.Thread(
            target=self._resume_worker,
            kwargs=dict(parent_dir=parent_dir, session_path=session_path,
                        job_map=job_map, cluster=cluster,
                        remote_base=remote_base, remote_exp_dir=remote_exp_dir,
                        poll_interval=poll_interval, n_total=n_total,
                        n_done_saved=n_done_saved, sess_version=sess_version,
                        time_limit=time_limit, time_limit_map=time_limit_map,
                        pipeline_script=pipeline_script,
                        pre_folders_map=pre_folders_map),
            daemon=True).start()

    def _resume_worker(self, parent_dir, session_path, job_map, cluster,
                       remote_base, remote_exp_dir, poll_interval,
                       n_total, n_done_saved, sess_version,
                       time_limit='', time_limit_map=None, pipeline_script='',
                       pre_folders_map=None):
        log = self._log
        try:
            # Clear any stale rows for this session's experiments so elapsed
            # clocks reset from the moment of resume, not from a previous session.
            self._clear_job_rows(list(job_map.values()))

            log('=' * 54, 'phase')
            log(f'Resuming session  ({sess_version})  —  querying SLURM…', 'phase')
            log('=' * 54, 'phase')
            self._start_phase('Resuming — querying SLURM…')

            active   = active_job_ids(cluster, set(job_map.keys()))
            finished = {jid: exp for jid, exp in job_map.items()
                        if jid not in active}
            log(f'  {len(active)} running  |  {len(finished)} already finished', 'normal')

            for jid, exp in job_map.items():
                status = 'running' if jid in active else 'completed'
                self._set_job_row(exp, jid, status, version=sess_version)

            n_done = n_done_saved
            for jid, exp_name in finished.items():
                local_exp  = parent_dir / exp_name
                remote_exp = f'{remote_exp_dir}/{exp_name}'
                log(f'  ↓  {exp_name}  (job {jid})', 'ok')
                pre = (pre_folders_map or {}).get(exp_name, None)
                if pre is None:
                    log('     (legacy session — copying all processed_on_* folders)', 'normal')
                rsync_new_outputs_down(cluster, remote_exp, local_exp, pre)
                n_done += 1
                self._set_job_row(exp_name, jid, 'completed', version=sess_version)
                self._phase_progress(
                    f'Catching up  —  {n_done}/{n_total} copied back', n_done, n_total)
                self._update_session_done(session_path, n_done)

            # Re-enable Run before handing off to the monitor thread
            self._set_submitting(False)
            threading.Thread(
                target=self._monitor_loop,
                kwargs=dict(
                    parent_dir=parent_dir, session_path=session_path,
                    job_map=job_map, pending=set(active), cluster=cluster,
                    remote_exp_dir=remote_exp_dir, poll_interval=poll_interval,
                    n_done=n_done, n_total=n_total, version=sess_version,
                    time_limit=time_limit, time_limit_map=time_limit_map,
                    pipeline_script=pipeline_script, remote_base=remote_base,
                    pre_folders_map=pre_folders_map),
                daemon=True).start()

        except Exception as exc:
            import traceback
            log(f'ERROR: {exc}', 'error')
            log(traceback.format_exc(), 'error')
        finally:
            self._set_submitting(False)   # always re-enable Run on catch-up failure

    def _monitor_loop(self, parent_dir, session_path, job_map, pending,
                      cluster, remote_exp_dir, poll_interval,
                      n_done, n_total, version=None,
                      time_limit='', time_limit_map=None,
                      pipeline_script='', remote_base='',
                      pre_folders_map=None):
        """Shared monitoring loop — runs as a daemon thread, never blocks the GUI.
        Multiple instances can run concurrently (one per active session).

        Auto-resubmit: when a job exits with TIMEOUT the wall-time is bumped
        by +2 h and the job is resubmitted automatically.
        """
        log            = self._log
        ver_lbl        = version or PIPELINE_VERSION
        time_limit_map = time_limit_map or {}

        self._phase_progress(f'Monitoring  ({ver_lbl})  —  {n_done}/{n_total} done',
                             n_done, n_total)

        while pending:
            time.sleep(poll_interval)

            active        = active_job_ids(cluster, pending)
            just_finished = pending - active

            for job_id in active:
                self._set_job_row(job_map[job_id], job_id, 'running', version=ver_lbl)

            resubmitted = {}   # new_job_id → exp_name (collected this iteration)

            for job_id in sorted(just_finished):
                exp_name   = job_map[job_id]
                remote_exp = f'{remote_exp_dir}/{exp_name}'
                local_exp  = parent_dir / exp_name
                ts         = datetime.now().strftime('%H:%M:%S')

                state = job_final_state(cluster, job_id)

                if state == 'TIMEOUT' and pipeline_script:
                    # ── Auto-resubmit with bumped wall-time ──────────────────
                    old_limit = time_limit_map.get(job_id, time_limit)
                    new_limit = _bump_time(old_limit)
                    log(f'[{ts}] ({ver_lbl}) job {job_id} TIMED OUT  '
                        f'→  resubmitting {exp_name}  --time={new_limit}', 'error')
                    self._set_job_row(exp_name, job_id, 'timeout', version=ver_lbl)

                    new_job_id = self._submit_one_job(
                        cluster, remote_base, remote_exp, pipeline_script, new_limit)
                    if new_job_id:
                        job_map[new_job_id]        = exp_name
                        time_limit_map[new_job_id] = new_limit
                        resubmitted[new_job_id]    = exp_name
                        log(f'         resubmitted as job {new_job_id}'
                            f'  (--time={new_limit})', 'ok')
                        self._set_job_row(exp_name, new_job_id, 'resubmitted',
                                          version=ver_lbl)
                        self._update_session_jobs(session_path, job_map,
                                                  time_limit_map)
                    else:
                        log(f'         resubmission failed — marking as failed', 'error')
                        n_done += 1
                        self._set_job_row(exp_name, job_id, 'failed', version=ver_lbl)
                        self._update_session_done(session_path, n_done)
                else:
                    # ── Normal finish (COMPLETED, FAILED, etc.) — copy back ──
                    log(f'[{ts}] ({ver_lbl}) job {job_id} {state}  ↓  {exp_name}', 'ok')
                    pre = (pre_folders_map or {}).get(exp_name, None)
                    if pre is None:
                        log('     (legacy session — copying all processed_on_* folders)', 'normal')
                    rsync_new_outputs_down(cluster, remote_exp, local_exp, pre)
                    n_done += 1
                    log(f'         copied back  ({n_done}/{n_total})', 'ok')
                    self._set_job_row(exp_name, job_id, 'completed', version=ver_lbl)
                    self._phase_progress(
                        f'Monitoring  ({ver_lbl})  —  {n_done}/{n_total} done',
                        n_done, n_total)
                    self._update_session_done(session_path, n_done)

            pending = active | set(resubmitted.keys())
            self._set_progress(
                f'({ver_lbl})  {n_done}/{n_total} done  |  {len(pending)} running')

        log('', 'normal')
        log(f'All jobs finished  ({ver_lbl}) — updating registry…', 'phase')
        final_reg = survey(parent_dir)
        reg_path  = parent_dir / REGISTRY_FILENAME
        with open(reg_path, 'w') as f:
            json.dump(final_reg, f, indent=2)
        log(f'Registry saved → {reg_path}', 'ok')
        self._finish_progress(f'Done  ({ver_lbl})  —  {n_done}/{n_total} completed')
        self._delete_session(session_path)
        self.after(0, lambda: self._bk_tab.refresh(parent_dir))

    def _do_cancel(self):
        """Cancel the active submission (phases 1-4 only)."""
        self._submit_cancel.set()
        self._log('--- Submission cancelled ---', 'error')

    def _run(self):
        if self._submitting:
            messagebox.showinfo('Busy', 'A submission is already in progress.')
            return
        p = Path(self._dir.get().strip())
        if not p.is_dir():
            messagebox.showerror('Error', f'Directory not found:\n{p}')
            return

        # Reload version.py so the field default always reflects the current file.
        _, _, current_pv = _reload_version()
        if not self._ver_var.get().strip():
            self._ver_var.set(current_pv)

        self._submit_cancel.clear()
        self._set_progress('')
        self._reset_progress()
        self._set_submitting(True)

        params = dict(
            parent_dir     = p,
            target_version = self._ver_var.get().strip() or None,
            run_all        = self._all_var.get(),
            cluster        = self._cluster_var.get().strip(),
            remote_base    = self._remote_var.get().strip().rstrip('/'),
            remote_exp_dir = (self._remote_exp_var.get().strip().rstrip('/') or
                              self._remote_var.get().strip().rstrip('/') + '/experiments'),
            poll_interval  = self._poll_var.get(),
            time_limit     = self._time_var.get().strip(),
        )
        threading.Thread(target=self._orchestrate, kwargs=params, daemon=True).start()

    # ── Background worker ─────────────────────────────────────────────────────

    def _orchestrate(self, parent_dir, target_version, run_all,
                     cluster, remote_base, remote_exp_dir, poll_interval,
                     time_limit=''):
        log   = self._log
        abort = self._submit_cancel

        try:
            # Reload version.py from disk so changes made after GUI startup are
            # picked up without restarting (new pipeline versions, manifest, etc.)
            VERSION_REGISTRY, CLUSTER_MANIFEST, _ = _reload_version()

            # ── Resolve pipeline script for this version ───────────────────────
            if target_version not in VERSION_REGISTRY:
                known = ', '.join(sorted(VERSION_REGISTRY))
                log(f'ERROR: version "{target_version}" is not registered in version.py.',
                    'error')
                log(f'  Known versions: {known}', 'error')
                log('  Add an entry to VERSION_REGISTRY in version.py first.', 'error')
                return
            pipeline_script = VERSION_REGISTRY[target_version]

            # ── Phase 1 — Select ──────────────────────────────────────────────
            self._start_phase('Phase 1 / 5  —  Selecting experiments…')
            log('=' * 54, 'phase')
            log('Phase 1 — Selecting experiments', 'phase')
            log('=' * 54, 'phase')
            log(f'  Version: {target_version}  →  {pipeline_script}', 'normal')
            registry = survey(parent_dir)
            selected = select_experiments(registry, target_version, run_all)
            if not selected:
                log('Nothing to do — all experiments already completed.', 'ok')
                log('Tick "Reprocess all" to resubmit.', 'normal')
                return
            log(f'{len(selected)} experiment(s) selected:', 'normal')
            for name, status in selected:
                log(f'  [{status}]  {name}', 'normal')
            to_process = [n for n, _ in selected]
            if abort.is_set(): return

            # ── Phase 2 — Sync all pipeline files to cluster ──────────────────
            self._start_phase('Phase 2 / 5  —  Syncing pipeline files to cluster…')
            log('', 'normal')
            log('=' * 54, 'phase')
            log('Phase 2 — Syncing pipeline files to cluster', 'phase')
            log('=' * 54, 'phase')

            base = Path(__file__).parent
            # Full file list: manifest + this version's pipeline script
            # (pipeline script may already be in CLUSTER_MANIFEST, dedup via dict)
            all_names = list(dict.fromkeys(CLUSTER_MANIFEST + [pipeline_script]))
            local_files, missing = [], []
            for fname in all_names:
                p = base / fname
                if p.exists():
                    local_files.append(p)
                else:
                    missing.append(fname)

            if missing:
                log(f'  WARNING: {len(missing)} file(s) not found locally — skipping:',
                    'error')
                for m in missing:
                    log(f'    {m}', 'error')

            log(f'  Syncing {len(local_files)} file(s)  '
                f'(rsync --update skips unchanged)', 'normal')
            for p in local_files:
                log(f'    {p.name}', 'normal')
            rsync_files_up(cluster, local_files, remote_base)
            log(f'  Sync complete → {remote_base}/', 'ok')
            if abort.is_set(): return

            # ── Phase 3 — Sync data ───────────────────────────────────────────
            self._start_phase('Phase 3 / 5  —  Syncing experiment data to cluster…')
            log('', 'normal')
            log('=' * 54, 'phase')
            log('Phase 3 — Syncing experiment data to cluster', 'phase')
            log('=' * 54, 'phase')
            ssh(cluster, f'mkdir -p {remote_exp_dir}')
            for exp_name in to_process:
                if abort.is_set(): return
                log(f'  ↑  {exp_name}', 'normal')
                ssh(cluster, f'mkdir -p {remote_exp_dir}/{exp_name}')
                rsync_up(cluster, parent_dir / exp_name,
                         f'{remote_exp_dir}/{exp_name}',
                         exclude=['processed_on_*'])

            # ── Phase 4 — Submit jobs ─────────────────────────────────────────
            self._start_phase('Phase 4 / 5  —  Submitting SLURM jobs…')
            log('', 'normal')
            log('=' * 54, 'phase')
            log('Phase 4 — Submitting SLURM jobs', 'phase')
            log('=' * 54, 'phase')

            # Snapshot pre-existing processed_on_* folders on the cluster for
            # each experiment so that copy-back only pulls the NEW folder created
            # by this run, not any older ones already present on the cluster.
            pre_folders_map: dict[str, set[str]] = {}
            for exp_name in to_process:
                remote_exp = f'{remote_exp_dir}/{exp_name}'
                pre_folders_map[exp_name] = list_remote_processed_folders(
                    cluster, remote_exp)
            log(f'  pre-run folder snapshot captured for {len(pre_folders_map)} experiment(s)',
                'normal')

            job_map        = {}
            time_limit_map = {}   # job_id → effective time limit used
            for exp_name in to_process:
                if abort.is_set(): return
                remote_exp = f'{remote_exp_dir}/{exp_name}'
                job_id = self._submit_one_job(
                    cluster, remote_base, remote_exp, pipeline_script, time_limit)
                if job_id is None:
                    log(f'  ERROR submitting {exp_name}', 'error')
                    self._set_job_row(exp_name, '—', 'failed')
                else:
                    job_map[job_id]        = exp_name
                    time_limit_map[job_id] = time_limit
                    log(f'  job {job_id}  →  {exp_name}'
                        + (f'  (--time={time_limit})' if time_limit else ''), 'ok')
                    self._set_job_row(exp_name, job_id, 'queued', version=target_version)
            if not job_map:
                log('No jobs submitted successfully.', 'error')
                return

            # Save session, then free the Run button before monitoring begins
            n_total      = len(job_map)
            session_path = self._save_session(
                parent_dir, job_map, cluster, remote_base, remote_exp_dir,
                poll_interval, n_total, pipeline_version=target_version,
                time_limit=time_limit, time_limit_map=time_limit_map,
                pipeline_script=pipeline_script,
                pre_folders_map=pre_folders_map)
            log(f'  Session saved → {session_path.name}', 'normal')

            # ── Phase 5 — Monitor (Run button re-enabled here) ─────────────────
            log('', 'normal')
            log('=' * 54, 'phase')
            log('Phase 5 — Monitoring + copying back as jobs finish', 'phase')
            log('=' * 54, 'phase')
            self._set_submitting(False)   # ← Run is now available for a new version

            threading.Thread(
                target=self._monitor_loop,
                kwargs=dict(
                    parent_dir=parent_dir, session_path=session_path,
                    job_map=job_map, pending=set(job_map.keys()),
                    cluster=cluster, remote_exp_dir=remote_exp_dir,
                    poll_interval=poll_interval,
                    n_done=0, n_total=n_total, version=target_version,
                    time_limit=time_limit, time_limit_map=time_limit_map,
                    pipeline_script=pipeline_script, remote_base=remote_base,
                    pre_folders_map=pre_folders_map),
                daemon=True).start()

        except Exception as exc:
            import traceback
            log(f'ERROR: {exc}', 'error')
            log(traceback.format_exc(), 'error')
            self._set_progress('Error — see log')
            self._reset_progress()
        finally:
            self._set_submitting(False)


# ── Application ───────────────────────────────────────────────────────────────

_GUI_CONFIG = Path(__file__).parent / 'gui_config.json'


def _load_gui_config():
    """Return the persisted GUI config dict, or {} if none exists."""
    try:
        with open(_GUI_CONFIG) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_gui_config(data: dict):
    """Merge data into the persisted GUI config."""
    cfg = _load_gui_config()
    cfg.update(data)
    try:
        with open(_GUI_CONFIG, 'w') as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f'SM Pipeline Manager  —  pipeline {PIPELINE_VERSION}')
        self.geometry('1000x780')
        self.resizable(True, True)

        cfg        = _load_gui_config()
        shared_dir = tk.StringVar(value=cfg.get('last_dir', ''))

        # Persist directory whenever it changes (both tabs write to the same var)
        shared_dir.trace_add('write',
            lambda *_: _save_gui_config({'last_dir': shared_dir.get().strip()}))

        nb     = ttk.Notebook(self)
        bk_tab = BookkeeperTab(nb, shared_dir)
        oc_tab = OrchestratorTab(nb, shared_dir, bk_tab)

        nb.add(bk_tab, text='   Bookkeeper   ')
        nb.add(oc_tab, text='   Orchestrator   ')
        nb.pack(fill='both', expand=True, padx=6, pady=6)


if __name__ == '__main__':
    App().mainloop()
