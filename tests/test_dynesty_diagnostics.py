"""Phase 6P tests for optional Dynesty periodic diagnostics."""

from types import SimpleNamespace

import numpy as np

import darksirens.inference.dynesty_diagnostics as diag


class _Results:
    def __init__(self, n=3):
        self.samples = np.zeros((n, 2))


class _Sampler:
    def __init__(self, results=None, error=None):
        self._results = results if results is not None else _Results()
        self._error = error

    @property
    def results(self):
        if self._error is not None:
            raise self._error
        return self._results


class _Figure:
    def __init__(self, kind, records):
        self.kind = kind
        self.records = records

    def savefig(self, path, **kwargs):
        self.records.append(("savefig", self.kind, path, kwargs))


class _Plotting:
    def __init__(self, records, *, run_error=None, trace_error=None):
        self.records = records
        self.run_error = run_error
        self.trace_error = trace_error

    def runplot(self, res, **kwargs):
        self.records.append(("runplot", kwargs))
        if self.run_error is not None:
            raise self.run_error
        return _Figure("run", self.records), object()

    def traceplot(self, res, **kwargs):
        self.records.append(("traceplot", kwargs))
        if self.trace_error is not None:
            raise self.trace_error
        return _Figure("trace", self.records), object()


class _Pyplot:
    def __init__(self, records):
        self.records = records

    def close(self, fig):
        self.records.append(("close", fig.kind))


def _install_plotting(monkeypatch, records, **kwargs):
    plotting = _Plotting(records, **kwargs)
    pyplot = _Pyplot(records)
    monkeypatch.setattr(diag, "_load_plotting", lambda: (plotting, pyplot))
    return plotting, pyplot


def test_diagnostics_dir_prefers_per_run_directory():
    assert diag.dynesty_diagnostics_dir(
        SimpleNamespace(save_path="/out", run_dir="/out/run_a")
    ) == "/out/run_a/dynesty_diagnostics"
    assert diag.dynesty_diagnostics_dir(
        SimpleNamespace(save_path="/out")
    ) == "/out/dynesty_diagnostics"
    assert diag.dynesty_diagnostics_dir(SimpleNamespace()) == "./dynesty_diagnostics"


def test_inconsistent_live_results_are_skipped(monkeypatch, tmp_path, capsys):
    records = []
    _install_plotting(monkeypatch, records)
    session = diag.DynestyDiagnosticsSession(
        _Sampler(error=RuntimeError("parallel arrays differ")),
        ["x", "y"],
        str(tmp_path),
    )
    session.write_once()
    assert capsys.readouterr().out == (
        "[dynesty diag] skipped an inconsistent concurrent read of "
        "sampler.results: parallel arrays differ\n"
    )
    assert records == []
    assert session._diag_index == 0


def test_too_few_samples_are_silent(monkeypatch, tmp_path, capsys):
    records = []
    _install_plotting(monkeypatch, records)
    session = diag.DynestyDiagnosticsSession(
        _Sampler(_Results(n=1)), ["x", "y"], str(tmp_path)
    )
    session.write_once()
    assert capsys.readouterr().out == ""
    assert records == []
    assert session._diag_index == 0


def test_successive_writes_preserve_filenames_and_plot_kwargs(
    monkeypatch, tmp_path, capsys
):
    records = []
    _install_plotting(monkeypatch, records)
    session = diag.DynestyDiagnosticsSession(
        _Sampler(_Results(n=3)), ["x", "y"], str(tmp_path)
    )
    session.write_once()
    session.write_once()
    assert capsys.readouterr().out == (
        f"[dynesty diag] wrote diagnostics #1 to {tmp_path}\n"
        f"[dynesty diag] wrote diagnostics #2 to {tmp_path}\n"
    )
    saves = [item for item in records if item[0] == "savefig"]
    assert [item[2] for item in saves] == [
        str(tmp_path / "runplot_0001.pdf"),
        str(tmp_path / "traceplot_0001.pdf"),
        str(tmp_path / "runplot_0002.pdf"),
        str(tmp_path / "traceplot_0002.pdf"),
    ]
    assert all(item[3] == {"bbox_inches": "tight"} for item in saves)
    assert records[0] == ("runplot", {"label_kwargs": {"fontsize": 10}})
    assert records[3] == (
        "traceplot",
        {
            "labels": ["x", "y"],
            "label_kwargs": {"fontsize": 8},
            "title_kwargs": {"fontsize": 8},
        },
    )


def test_plot_failures_are_contained_independently(monkeypatch, tmp_path, capsys):
    records = []
    _install_plotting(
        monkeypatch,
        records,
        run_error=RuntimeError("run broke"),
        trace_error=RuntimeError("trace broke"),
    )
    session = diag.DynestyDiagnosticsSession(
        _Sampler(_Results(n=3)), ["x", "y"], str(tmp_path)
    )
    session.write_once()
    assert capsys.readouterr().out == (
        "[dynesty diag] runplot failed: run broke\n"
        "[dynesty diag] traceplot failed: trace broke\n"
        f"[dynesty diag] wrote diagnostics #1 to {tmp_path}\n"
    )
    assert [item[0] for item in records] == ["runplot", "traceplot"]


class _LoopEvent:
    def __init__(self):
        self.waits = []
        self._stopped = False

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if len(self.waits) >= 2:
            self._stopped = True

    def is_set(self):
        return self._stopped

    def set(self):
        self._stopped = True


def test_thread_loop_contains_uncaught_writer_failure(tmp_path, capsys):
    session = diag.DynestyDiagnosticsSession(
        _Sampler(_Results(n=3)), ["x"], str(tmp_path), interval_seconds=600
    )
    event = _LoopEvent()
    session._stop_diag = event

    def broken_writer():
        raise RuntimeError("outer failure")

    session.write_once = broken_writer
    session._thread_fn()
    assert event.waits == [600, 600]
    assert capsys.readouterr().out == (
        "[dynesty diag] diagnostics pass failed: outer failure\n"
    )


def test_start_and_stop_use_frozen_daemon_and_join_timeout(
    monkeypatch, tmp_path, capsys
):
    records = []
    _install_plotting(monkeypatch, records)

    class Event:
        def __init__(self):
            self.set_called = False

        def set(self):
            self.set_called = True

        def wait(self, timeout=None):
            raise AssertionError("fake thread must not execute target")

        def is_set(self):
            return self.set_called

    class Thread:
        def __init__(self, target, daemon):
            self.target = target
            self.daemon = daemon
            self.started = False
            self.join_timeout = None

        def start(self):
            self.started = True

        def join(self, timeout=None):
            self.join_timeout = timeout

    event = Event()
    made = []
    monkeypatch.setattr(diag.threading, "Event", lambda: event)
    monkeypatch.setattr(
        diag.threading,
        "Thread",
        lambda target, daemon: made.append(Thread(target, daemon)) or made[-1],
    )

    session = diag.DynestyDiagnosticsSession(
        _Sampler(_Results(n=3)), ["x", "y"], str(tmp_path)
    ).start()
    assert capsys.readouterr().out == (
        f"[*] Diagnostic plots enabled — writing to {tmp_path}/ every 10 min.\n"
    )
    assert len(made) == 1
    assert made[0].daemon is True
    assert made[0].started is True

    session.stop()
    assert event.set_called is True
    assert made[0].join_timeout == 120
