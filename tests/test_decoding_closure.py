import json
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/decoding-review-closure-20261006-v1/calibrated"
spec = importlib.util.spec_from_file_location("closure_verify", ROOT / "scripts/verify_decoding_closure.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_full_ratings_and_balanced_arms():
    ratings = json.loads((OUT / "ratings.json").read_text(encoding="utf-8"))
    assert len(ratings) == 270
    for arm in ("control", "candidate"):
        selected = [r for r in ratings if r["arm"] == arm]
        assert len(selected) == len({r["id"] for r in selected}) == 135
        assert all(all(r[k] is not None for k in ("intent_score", "helpfulness_score", "fluency_score", "repetition", "unsupported_claim")) for r in selected)


def test_same_text_cannot_receive_different_scores():
    rows = json.loads((OUT / "ratings.json").read_text(encoding="utf-8"))
    by_source = {}
    for r in rows:
        key = r["source_review_id"]
        values = tuple(r[k] for k in ("intent_score", "helpfulness_score", "fluency_score", "repetition", "unsupported_claim"))
        assert by_source.setdefault(key, values) == values


def test_entity_inventory_preserves_leading_zero_and_concrete_type():
    values = m.input_entities("download invoice #00108 from a platinum account")
    assert [v["text"] for v in values] == ["00108", "platinum"]
    assert [v["role"] for v in values] == ["invoice_id", "account_type"]


def test_initial_judgments_not_silently_overwritten():
    raw = json.loads((OUT.parent / "ratings.json").read_text(encoding="utf-8"))
    calibrated = json.loads((OUT / "ratings.json").read_text(encoding="utf-8"))
    assert raw != calibrated
    assert (OUT.parent / "final-freeze.json").exists()
    assert (OUT / "calibration-changes.json").exists()


def test_final_veto_and_paired_repetition_counts():
    s = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert sum(s["paired_repetition"].values()) == 135
    assert not s["adoption_approved"] and s["per_id_veto_ids"]
    assert s["metrics_sums"]["all135"]["control"]["repetition"] == 63
    assert s["metrics_sums"]["all135"]["candidate"]["repetition"] == 65
