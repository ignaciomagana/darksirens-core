"""The repository states no H0 value measured on real gravitational-wave events.

The analyses that use this library are blinded to the real-data Hubble
constant, so the documentation and the source comments describe real-event
effects by their direction and their size against the posterior width only.
Mock, toy and clean-room numbers stay: they are what the documentation
measures the estimators on.

The check looks for an H0 value -- a number given in km/s/Mpc (plain or LaTeX
units), or a bare ``H0 = NN`` -- whose passage names a GWTC event set, a GW
event, or real data. A passage is the run of non-blank lines around the value
(a Markdown paragraph, a comment block, a docstring), looked at both before
and after the value. A failure prints the file and line only, never the line
itself.

The planted examples below are assembled from pieces, so this file itself
contains no statement the scan would flag.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# The unit, plain (km/s/Mpc, km s^-1 Mpc^-1, km/sec/Mpc) or LaTeX
# (km\,s^{-1}\,Mpc^{-1}, \mathrm{km\,s^{-1}\,Mpc^{-1}}, km s$^{-1}$ Mpc$^{-1}$).
_TEX = r"(?:\\(?:mathrm|textrm|text|rm)\s*\{?\s*)?"
_UNIT = (
    _TEX + r"km\}?[^A-Za-z\n]{0,10}"
    + _TEX + r"s(?:ec)?\}?[^A-Za-z\n]{0,14}"
    + _TEX + r"Mpc"
)
# A number followed, on the same line, by the unit.
_H0_NUMBER = re.compile(r"[0-9][^A-Za-z\n]{0,24}" + _UNIT)
# A bare two- or three-digit value given to H0 with no unit.
_H0_BARE = re.compile(
    r"\bH_?\{?0\}?\s*(?:=|\\approx|\\simeq|≈|~)\s*\$?\s*[0-9]{2,3}(?:\.[0-9]+)?\b"
)
# What marks a passage as being about real events, not a mock or a toy. The
# catalog name is matched case-sensitively, so a package name such as
# ``darksirens-desi-gwtc4`` does not count.
_MARKERS = (
    re.compile(r"GWTC-[0-9]"),
    re.compile(r"\bGW[0-9]{6}\b"),
    re.compile(
        r"\breal[- ](?:data|events?|GW|catalog|posteriors?)\b", re.IGNORECASE
    ),
)
# How far the passage reaches from the value, within its run of non-blank lines.
_BACK = 8
_FORWARD = 4
_SUFFIXES = {".md", ".rst", ".txt", ".py", ".yml", ".yaml", ".toml", ".cfg", ".json"}
_SKIP_DIRS = {".git", ".venv", "venv", "build", "dist", "node_modules", "__pycache__"}


def _text_files() -> list[Path]:
    """Every tracked text file; every text file when there is no git checkout."""
    if (ROOT / ".git").exists():
        try:
            listing = subprocess.run(
                ["git", "-C", str(ROOT), "ls-files", "-z"],
                capture_output=True,
                check=True,
            ).stdout.decode("utf-8", errors="replace")
        except (OSError, subprocess.CalledProcessError):
            listing = None
        if listing is not None:
            paths = (ROOT / name for name in listing.split("\0") if name)
            return sorted(p for p in paths if p.suffix in _SUFFIXES and p.is_file())
    return sorted(
        p
        for p in ROOT.rglob("*")
        if p.suffix in _SUFFIXES
        and p.is_file()
        and not _SKIP_DIRS.intersection(p.relative_to(ROOT).parts)
        and not any(part.endswith(".egg-info") for part in p.parts)
    )


def _passage(lines: list[str], index: int) -> list[str]:
    """The line and its neighbours in the same run of non-blank lines."""
    lo = index
    while lo > 0 and index - lo < _BACK and lines[lo - 1].strip():
        lo -= 1
    hi = index
    while hi < len(lines) - 1 and hi - index < _FORWARD and lines[hi + 1].strip():
        hi += 1
    return lines[lo : hi + 1]


def _real_event_h0_lines(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    hits = []
    for i, line in enumerate(lines):
        if not (_H0_NUMBER.search(line) or _H0_BARE.search(line)):
            continue
        passage = _passage(lines, i)
        if any(m.search(item) for m in _MARKERS for item in passage):
            hits.append(i + 1)
    return hits


def test_files_are_scanned():
    names = {p.relative_to(ROOT).as_posix() for p in _text_files()}
    assert "MIGRATION.md" in names
    assert "src/darksirens/population/registry.py" in names
    assert "tests/test_no_real_event_h0_values.py" in names
    if (ROOT / ".git").exists():
        assert "tests/reference/legacy/README.md" in names
        assert any(n.startswith(".github/workflows/") for n in names)


@pytest.mark.parametrize(
    "relative",
    ["MIGRATION.md", "src/darksirens/population/registry.py"],
)
def test_effective_spin_fix_states_no_h0_number(relative):
    # The note on the corrected effective-spin pair of the GWTC fixed
    # population gives the effect on H0 without a value.
    assert _real_event_h0_lines(ROOT / relative) == [], relative


def test_no_tracked_text_states_a_real_event_h0_value():
    found = []
    for path in _text_files():
        for number in _real_event_h0_lines(path):
            found.append(f"{path.relative_to(ROOT).as_posix()}:{number}")
    assert found == [], "real-event H0 value stated at " + ", ".join(found)


# Pieces for the planted examples (kept apart so this file stays clean).
_CAT = "GW" + "TC-4"
_EVENT = "GW" + "190521"
_REAL_DATA = "re" + "al data"
_KM = "k" + "m"
_KMS = _KM + "/s/Mpc"
_H0_IS = "H" + "0 = 10.0"


def _plant(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return _real_event_h0_lines(path)


def test_the_scan_sees_a_marker_before_the_value(tmp_path):
    md = (
        f"On the {_CAT} events the fixed-population H0\n"
        f"is 1.0 {_KMS} lower.\n\n"
        f"On the mock the offset is 1.0 {_KMS}.\n"
    )
    assert _plant(tmp_path, "a.md", md) == [2]
    py = f"# On the {_CAT} events H0 moves\n# by about -1.0 {_KMS}.\nx = 1\n"
    assert _plant(tmp_path, "a.py", py) == [2]


def test_the_scan_sees_a_marker_after_the_value(tmp_path):
    md = f"The fixed-population H0 moves by 1.0 {_KMS}\non the {_CAT} events.\n"
    assert _plant(tmp_path, "b.md", md) == [1]


def test_the_scan_sees_a_docstring(tmp_path):
    py = (
        "def f():\n"
        '    """Fit the spin pair.\n'
        "\n"
        f"    On {_EVENT} and the other events the\n"
        f"    posterior sits 1.0 {_KMS} lower.\n"
        '    """\n'
    )
    assert _plant(tmp_path, "c.py", py) == [5]


def test_the_scan_sees_latex_units_and_bare_values(tmp_path):
    # "KM" stands in for the unit's first word until the text is planted.
    tex_units = (
        r"1.0~\mathrm{KM\,s^{-1}\,Mpc^{-1}}",
        r"1.0 KM\,s$^{-1}$\,Mpc$^{-1}$",
        "1.0 KM s^-1 Mpc^-1",
        "1.0 KM/sec/Mpc",
    )
    for k, unit in enumerate(tex_units):
        unit = unit.replace("KM", _KM)
        md = f"On the {_CAT} events H0 is lower by {unit}.\n"
        assert _plant(tmp_path, f"d{k}.md", md) == [1], unit
    bare = f"On {_REAL_DATA} the fixed-population run gives {_H0_IS}.\n"
    assert _plant(tmp_path, "e.md", bare) == [1]


def test_the_scan_leaves_mock_numbers_alone(tmp_path):
    md = (
        f"On the mock catalog the offset is 1.0 {_KMS} and {_H0_IS}.\n\n"
        f"The {_CAT} events are read from the gwcat store.\n"
    )
    assert _plant(tmp_path, "f.md", md) == []
