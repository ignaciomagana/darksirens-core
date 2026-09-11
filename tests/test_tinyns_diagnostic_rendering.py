"""Phase 6L tests for TinyNS diagnostic rendering."""

from darksirens.inference.tinyns_output import print_tinyns_diagnostics


def test_empty_diagnostics_print_nothing(capsys):
    assert print_tinyns_diagnostics({}) is None
    assert capsys.readouterr().out == ""


def test_full_diagnostic_render_is_frozen(capsys):
    diag = {
        "success": True,
        "message": "converged",
        "logz": -12.5,
        "logzerr": 0.125,
        "niter": 10,
        "ncall": 250,
        "seconds": 5.0,
        "niter_per_sec": 2.0,
        "ncall_per_sec": 50.0,
        "calls_per_iter": 25.0,
        "final_delta_logz": 0.01,
        "live_weight_fraction": 0.2,
        "posterior_ess": 100.0,
        "replacement_mean_batches": 1.5,
        "replacement_max_batches": 4,
        "replacement_failures": 2,
        "replacement_rescue_used": True,
        "insertion_rank_mean_z": -0.3,
        "insertion_rank_std_ratio": 1.1,
    }
    print_tinyns_diagnostics(diag)
    assert capsys.readouterr().out == (
        "TinyNS Diagnostics\n"
        "------------------\n"
        "success: true\n"
        "message: converged\n"
        "logZ: -12.5 ± 0.125\n"
        "niter: 10\n"
        "ncall: 250\n"
        "wall time: 5 s\n"
        "niter/sec: 2\n"
        "ncall/sec: 50\n"
        "calls/iter: 25\n"
        "final dlogz: 0.01\n"
        "live weight fraction: 0.2\n"
        "posterior ESS: 100\n"
        "replacement mean batches: 1.5\n"
        "replacement max batches: 4\n"
        "replacement failures: 2\n"
        "replacement rescue used: yes\n"
        "insertion rank mean z: -0.3\n"
        "insertion rank std ratio: 1.1\n"
    )


def test_sparse_render_omits_missing_fields_and_formats_false(capsys):
    print_tinyns_diagnostics(
        {
            "success": False,
            "logz": -1,
            "replacement_rescue_used": False,
            "unknown": "not rendered",
        }
    )
    assert capsys.readouterr().out == (
        "TinyNS Diagnostics\n"
        "------------------\n"
        "success: false\n"
        "logZ: -1\n"
        "replacement rescue used: no\n"
    )


def test_none_values_are_treated_as_absent(capsys):
    print_tinyns_diagnostics(
        {
            "success": None,
            "message": None,
            "logz": -2.0,
            "logzerr": None,
            "seconds": None,
        }
    )
    assert capsys.readouterr().out == (
        "TinyNS Diagnostics\n"
        "------------------\n"
        "logZ: -2\n"
    )
