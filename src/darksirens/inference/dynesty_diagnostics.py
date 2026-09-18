"""Optional periodic diagnostics for a running Dynesty sampler.

The sampling adapter owns inference semantics; this module owns only the frozen
``--dynesty_diagnostics`` side effect.  Dynesty plotting and Matplotlib remain
lazy so importing the core inference surface does not pull plotting backends.
"""

from __future__ import annotations

import os
import threading


def dynesty_diagnostics_dir(opts):
    """Return the per-run diagnostics directory, preserving legacy fallback."""
    root = getattr(opts, "run_dir", None) or getattr(opts, "save_path", ".")
    return os.path.join(root, "dynesty_diagnostics")


def _load_plotting():
    import dynesty.plotting as dyplot

    try:
        import matplotlib
    except ImportError as exc:
        raise ImportError(
            "dynesty diagnostics need matplotlib; install the 'dynesty' extra "
            "(pip install 'darksirens[dynesty]')"
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return dyplot, plt


class DynestyDiagnosticsSession:
    """Manage the frozen ten-minute Dynesty plotting daemon."""

    def __init__(self, sampler, labels, out_dir, *, interval_seconds=600):
        self.sampler = sampler
        self.labels = labels
        self.out_dir = out_dir
        self.interval_seconds = interval_seconds
        self._diag_index = 0
        self._stop_diag = threading.Event()
        self._thread = None
        self._dyplot = None
        self._plt = None

    def _ensure_plotting(self):
        if self._dyplot is None or self._plt is None:
            self._dyplot, self._plt = _load_plotting()

    def write_once(self):
        """Write one run/trace diagnostic pair, or skip a torn live read."""
        try:
            res = self.sampler.results
            n_samples = len(res.samples)
        except Exception as exc:
            print(
                "[dynesty diag] skipped an inconsistent concurrent read "
                f"of sampler.results: {exc}",
                flush=True,
            )
            return
        if n_samples < 2:
            return

        self._diag_index += 1
        idx = self._diag_index
        os.makedirs(self.out_dir, exist_ok=True)
        self._ensure_plotting()

        try:
            fig, _ = self._dyplot.runplot(
                res,
                label_kwargs={"fontsize": 10},
            )
            fig.savefig(
                os.path.join(self.out_dir, f"runplot_{idx:04d}.pdf"),
                bbox_inches="tight",
            )
            self._plt.close(fig)
        except Exception as exc:
            print(f"[dynesty diag] runplot failed: {exc}", flush=True)

        try:
            fig, _ = self._dyplot.traceplot(
                res,
                labels=self.labels,
                label_kwargs={"fontsize": 8},
                title_kwargs={"fontsize": 8},
            )
            fig.savefig(
                os.path.join(self.out_dir, f"traceplot_{idx:04d}.pdf"),
                bbox_inches="tight",
            )
            self._plt.close(fig)
        except Exception as exc:
            print(f"[dynesty diag] traceplot failed: {exc}", flush=True)

        print(
            f"[dynesty diag] wrote diagnostics #{idx} to {self.out_dir}",
            flush=True,
        )

    def _thread_fn(self):
        # Wait one full interval before the first plot so Dynesty has real
        # samples.  Nothing raised by a diagnostics pass may kill this daemon.
        self._stop_diag.wait(timeout=self.interval_seconds)
        while not self._stop_diag.is_set():
            try:
                self.write_once()
            except Exception as exc:
                print(
                    f"[dynesty diag] diagnostics pass failed: {exc}",
                    flush=True,
                )
            self._stop_diag.wait(timeout=self.interval_seconds)

    def start(self):
        """Start the diagnostics daemon and return this session."""
        # Match the old branch's eager plotting-backend validation once the
        # option is enabled, while keeping ordinary adapter imports light.
        self._ensure_plotting()
        self._thread = threading.Thread(target=self._thread_fn, daemon=True)
        self._thread.start()
        print(
            f"[*] Diagnostic plots enabled — writing to {self.out_dir}/ "
            "every 10 min.",
            flush=True,
        )
        return self

    def stop(self):
        """Stop and join the daemon using the frozen shutdown timeout."""
        self._stop_diag.set()
        if self._thread is not None:
            self._thread.join(timeout=120)


def start_dynesty_diagnostics(sampler, labels, opts):
    """Create and start the frozen periodic diagnostics session."""
    return DynestyDiagnosticsSession(
        sampler,
        labels,
        dynesty_diagnostics_dir(opts),
        interval_seconds=600,
    ).start()


__all__ = [
    "DynestyDiagnosticsSession",
    "dynesty_diagnostics_dir",
    "start_dynesty_diagnostics",
]
