#!/usr/bin/env python3
"""Separate-process parity probe for Phase 6P Dynesty diagnostics."""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

OUT_DIR = "/tmp/darksirens-phase6p-diagnostics"
LABELS = ["x", "y"]


class _Results:
    def __init__(self, n=3):
        self.samples = np.zeros((n, 2))


class _Sampler:
    def __init__(self, n=3, *, error=None):
        self._results = _Results(n)
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
        self.records.append(["savefig", self.kind, path, kwargs])


class _Plotting:
    def __init__(self, records, *, run_error=None, trace_error=None):
        self.records = records
        self.run_error = run_error
        self.trace_error = trace_error

    def runplot(self, res, **kwargs):
        self.records.append(["runplot", kwargs])
        if self.run_error:
            raise RuntimeError(self.run_error)
        return _Figure("run", self.records), object()

    def traceplot(self, res, **kwargs):
        self.records.append(["traceplot", kwargs])
        if self.trace_error:
            raise RuntimeError(self.trace_error)
        return _Figure("trace", self.records), object()


class _Pyplot:
    def __init__(self, records):
        self.records = records

    def close(self, fig):
        self.records.append(["close", fig.kind])


class _LoopEvent:
    def __init__(self):
        self.waits = []
        self.stopped = False

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if len(self.waits) >= 2:
            self.stopped = True

    def is_set(self):
        return self.stopped

    def set(self):
        self.stopped = True


def _stdout(call):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        call()
    return stream.getvalue()


def _legacy_functions(root: Path):
    source = root / "darksirens" / "inference" / "sampling.py"
    tree = ast.parse(source.read_text(), filename=str(source))

    top = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "dynesty_diagnostics_dir"
    ]
    runs = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_sampler"
    ]
    if len(top) != 1 or len(runs) != 1:
        raise RuntimeError("frozen diagnostics definitions were not unique")

    nested = {
        node.name: node
        for node in ast.walk(runs[0])
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_write_dynesty_diagnostics", "_diag_thread_fn"}
    }
    if set(nested) != {"_write_dynesty_diagnostics", "_diag_thread_fn"}:
        raise RuntimeError("frozen nested diagnostics functions were not found")

    def compile_one(node, namespace):
        module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        exec(compile(module, str(source), "exec"), namespace)
        return namespace[node.name]

    path_fn = compile_one(top[0], {"os": os})
    return source, path_fn, nested, compile_one


def _legacy_writer(nested, compile_one, sampler, plotting, pyplot, *, calls=1):
    namespace = {
        "os": os,
        "dyplot": plotting,
        "plt": pyplot,
        "labels": LABELS,
        "diag_dir": OUT_DIR,
        "_diag_index": [0],
    }
    writer = compile_one(nested["_write_dynesty_diagnostics"], namespace)
    return _stdout(lambda: [writer(sampler) for _ in range(calls)]), namespace


def _candidate_writer(module, sampler, plotting, pyplot, *, calls=1):
    session = module.DynestyDiagnosticsSession(sampler, LABELS, OUT_DIR)
    session._dyplot = plotting
    session._plt = pyplot
    stdout = _stdout(lambda: [session.write_once() for _ in range(calls)])
    return stdout, session


def _writer_cases(implementation, legacy=None):
    import darksirens.inference.dynesty_diagnostics as candidate

    cases = {
        "success_twice": dict(n=3, calls=2),
        "short": dict(n=1, calls=1),
        "inconsistent": dict(n=3, calls=1, read_error="parallel arrays differ"),
        "runplot_failure": dict(n=3, calls=1, run_error="run broke"),
        "traceplot_failure": dict(n=3, calls=1, trace_error="trace broke"),
    }
    out = {}
    for name, spec in cases.items():
        records = []
        sampler = _Sampler(
            spec["n"],
            error=(
                RuntimeError(spec["read_error"])
                if spec.get("read_error")
                else None
            ),
        )
        plotting = _Plotting(
            records,
            run_error=spec.get("run_error"),
            trace_error=spec.get("trace_error"),
        )
        pyplot = _Pyplot(records)
        if implementation == "legacy":
            _source, _path, nested, compile_one = legacy
            stdout, state = _legacy_writer(
                nested,
                compile_one,
                sampler,
                plotting,
                pyplot,
                calls=spec["calls"],
            )
            index = state["_diag_index"][0]
        else:
            stdout, session = _candidate_writer(
                candidate,
                sampler,
                plotting,
                pyplot,
                calls=spec["calls"],
            )
            index = session._diag_index
        out[name] = {"stdout": stdout, "records": records, "index": index}
    return out


def _thread_cases(implementation, legacy=None):
    import darksirens.inference.dynesty_diagnostics as candidate

    out = {}
    for name, raises in (("success", False), ("writer_failure", True)):
        event = _LoopEvent()
        records = []

        def writer(_sampler=None):
            records.append("write")
            if raises:
                raise RuntimeError("outer failure")

        if implementation == "legacy":
            _source, _path, nested, compile_one = legacy
            namespace = {
                "_stop_diag": event,
                "diag_interval": 600,
                "_write_dynesty_diagnostics": writer,
                "sampler": object(),
            }
            thread_fn = compile_one(nested["_diag_thread_fn"], namespace)
            stdout = _stdout(thread_fn)
        else:
            session = candidate.DynestyDiagnosticsSession(
                object(), LABELS, OUT_DIR, interval_seconds=600
            )
            session._stop_diag = event
            session.write_once = lambda: writer()
            stdout = _stdout(session._thread_fn)

        out[name] = {
            "stdout": stdout,
            "records": records,
            "waits": event.waits,
            "stopped": event.stopped,
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("legacy", "candidate"), required=True)
    parser.add_argument("--legacy-root")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    legacy = None
    if args.implementation == "legacy":
        if not args.legacy_root:
            raise SystemExit("--legacy-root required for legacy")
        legacy = _legacy_functions(Path(args.legacy_root))
        path_fn = legacy[1]
    else:
        import darksirens.inference.dynesty_diagnostics as candidate

        path_fn = candidate.dynesty_diagnostics_dir

    paths = {
        "run_dir": path_fn(SimpleNamespace(save_path="/out", run_dir="/out/run_a")),
        "save_path": path_fn(SimpleNamespace(save_path="/out")),
        "default": path_fn(SimpleNamespace()),
    }
    behavior = {
        "paths": paths,
        "writers": _writer_cases(args.implementation, legacy),
        "threads": _thread_cases(args.implementation, legacy),
    }
    Path(args.out).write_text(json.dumps(behavior, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
