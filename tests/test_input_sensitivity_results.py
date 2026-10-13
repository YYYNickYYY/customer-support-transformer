import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/input-sensitivity-20261006-v1"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def test_complete_explicit_assessment_and_nonapplicable_nulls():
    plan = read("plan.json")
    cases = {c["id"]: c for g in plan["groups"] for c in g["cases"]}
    ratings = read("assessments.json")
    assert len(ratings) == len({r["id"] for r in ratings}) == len(cases) == 22
    for row in ratings:
        assert type(row["action_match"]) is bool
        assert row["evidence"] and row["notes"]
        if not cases[row["id"]].get("negation_required"):
            assert row["negation_respected"] is None
        if "required_id" not in cases[row["id"]]:
            assert row["required_id_copied"] is None


def test_controls_not_mistaken_for_semantic_success():
    s = read("assessment-summary.json")
    assert s["control_reproducible"]
    assert s["negation_probes"] == {"n": 2, "respected": 0}
    assert s["explicit_copy_probes"] == {"n": 4, "correct": 0}
    invoice = next(g for g in s["groups"] if g["group"] == "G06")
    assert invoice["first_pair_exact_output"] and not invoice["passed"]
    assert invoice["first_pair_first_step_top5_differs"]
    assert not s["training_started"] and s["optimizer_updates"] == 0


def test_assessment_artifacts_and_implementation_hashes_still_frozen():
    for name in ("freeze.json", "generation-freeze.json", "assessment-freeze.json"):
        for path, expected in read(name)["sha256"].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
