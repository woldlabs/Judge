import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "judge_vision_risk_delta",
    Path(__file__).resolve().parents[1] / "scripts" / "vision_risk_delta.py",
)
_mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_mod)
map_paths = _mod.map_paths
render_comment = _mod.render_comment


def test_maps_detector_and_reporting_paths():
    mapped = map_paths(
        [
            "judge/core/detectors/video.py",
            "judge/reporting/evidence_pack.py",
            "docs/QUICKSTART.md",
            "LICENSE",
        ]
    )
    themes = {row["theme"] for row in mapped["themes"]}
    assert "Architecture boundaries" in themes
    assert any("Rift" in t for t in themes)
    assert "Success metrics (docs / operator path)" in themes
    assert "LICENSE" in mapped["unmatched"]
    assert "needs-vision-review" in mapped["labels"]


def test_comment_links_are_absolute():
    mapped = map_paths(["VISION.md"])
    body = render_comment(mapped)
    assert "<!-- judge-vision-risk-delta -->" in body
    assert "https://github.com/woldlabs/Judge/blob/main/VISION.md" in body
    assert "https://github.com/woldlabs/Judge/blob/main/docs/TRIAGE.md" in body
    assert "../blob/main" not in body


def test_noop_comment_uses_absolute_triage_link():
    body = render_comment(map_paths(["LICENSE"]))
    assert "https://github.com/woldlabs/Judge/blob/main/docs/TRIAGE.md" in body
    assert "../blob/" not in body
