"""The repository states no H0 value measured on real gravitational-wave events.

The analyses that use this library are blinded to the real-data Hubble
constant, so the documentation and the source comments describe real-event
effects by their direction and their size against the posterior width only.
Mock, toy and clean-room numbers stay: they are what the documentation
measures the estimators on.

The check looks for a number given in km/s/Mpc whose paragraph (in a Markdown
file) or comment block (in a Python file) names a GWTC event set or real
data. A failure prints the file and line only, never the line itself.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# A number followed, on the same line, by km/s/Mpc or km s^-1 Mpc^-1.
_H0_NUMBER = re.compile(
    r"[0-9](?:[^A-Za-z\n]{0,24})(?:km\s*/\s*s\s*/\s*Mpc|km\s*s\^?\{?-1\}?\s*Mpc)"
)
# What marks a passage as being about real events, not a mock or a toy. The
# catalog name is matched case-sensitively, so a package name such as
# ``darksirens-desi-gwtc4`` does not count.
_GWTC = re.compile(r"GWTC-[0-9]")
_REAL = re.compile(r"\breal[- ](?:data|event)", re.IGNORECASE)
# How far back within one paragraph or comment block the marker is looked for.
_WINDOW = 4


def _text_files() -> list[Path]:
    files = sorted(ROOT.glob("*.md"))
    for sub in ("src", "tools", "examples"):
        base = ROOT / sub
        if base.is_dir():
            files += sorted(base.rglob("*.py"))
            files += sorted(base.rglob("*.md"))
    return files


def _passage_before(lines: list[str], index: int, python: bool) -> list[str]:
    """The line and up to ``_WINDOW`` lines before it in the same passage."""
    out = [lines[index]]
    for j in range(index - 1, max(-1, index - 1 - _WINDOW), -1):
        line = lines[j]
        if not line.strip():
            break
        if python and not line.lstrip().startswith("#"):
            break
        out.append(line)
    return out


def _real_event_h0_lines(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    python = path.suffix == ".py"
    hits = []
    for i, line in enumerate(lines):
        if not _H0_NUMBER.search(line):
            continue
        passage = _passage_before(lines, i, python)
        if any(_GWTC.search(item) or _REAL.search(item) for item in passage):
            hits.append(i + 1)
    return hits


def test_files_are_scanned():
    names = {p.relative_to(ROOT).as_posix() for p in _text_files()}
    assert "MIGRATION.md" in names
    assert "src/darksirens/population/registry.py" in names


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


def test_the_scan_sees_a_planted_statement(tmp_path):
    planted = tmp_path / "planted.md"
    planted.write_text(
        "On the GWTC-4 events the fixed-population H0\n"
        "is 1.0 km/s/Mpc lower.\n\n"
        "On the mock the offset is 1.0 km/s/Mpc.\n",
        encoding="utf-8",
    )
    assert _real_event_h0_lines(planted) == [2]
    comment = tmp_path / "planted.py"
    comment.write_text(
        "# On the GWTC-4 events H0 moves\n# by about -1.0 km/s/Mpc.\nx = 1\n",
        encoding="utf-8",
    )
    assert _real_event_h0_lines(comment) == [2]
