"""Freeze a preregistration only; no torch, model loading or inference."""

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/decoding-repetition-preregister-20261006-v1"


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with (ROOT / path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def register():
    plan = read(OUT.relative_to(ROOT) / "plan.json")
    assert plan["status"] == "preregistered_not_run"
    assert (plan["control"], plan["candidate"]) == (0, 4)
    assert plan["results"] is None and plan["generation_runs"] == 0
    config = read(plan["baseline_config"])
    for key in ("device", "cpu_threads", "seed", "max_new_tokens"):
        assert config[key] == plan["fixed"][key]
    assert config["model_config"]["max_length"] == 512
    samples = read(plan["selection"])["samples"]
    ids = [s["id"] for s in samples]
    assert len(ids) == len(set(ids)) == 135
    assert Counter(s["cohort"] for s in samples) == plan["cohorts"]
    intents = Counter(s["intent"] for s in samples)
    assert len(intents) == 27 and set(intents.values()) == {5}
    replies = [json.loads(line) for line in (ROOT / plan["baseline_replies"]).read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in replies] == ids
    ratings = [r for r in read(plan["baseline_scores"])["ratings"] if r["model"] == "fifth"]
    assert len(ratings) == 135 and {r["id"] for r in ratings} == set(ids)
    assert sum(r["repetition"] for r in ratings) == 63
    assert sum(r["unsupported_claim"] for r in ratings) == 43
    assert digest(config["checkpoint"]) == config["checkpoint_sha256"]
    assert digest(plan["selection"]) == config["selection_sha256"]
    assert digest("artifacts/tokenizer/bitext-bpe-v1/tokenizer.json") == config["model_config"]["tokenizer_revision"]
    paths = [str(OUT.relative_to(ROOT) / "plan.json"), plan["protocol"],
             plan["baseline_config"], plan["selection"], plan["baseline_replies"],
             plan["baseline_scores"], plan["secondary_selection"], config["checkpoint"],
             *plan["rubrics"].values(), "scripts/register_decoding_comparison.py",
             "outputs/dev-expanded-fifth-20261005/ai-expert-ratings.json",
             "docs/fifth-generation-diagnostic-20261006.md",
             "outputs/fifth-generation-diagnostic-20261006-v1.1/analysis-freeze.json",
             "outputs/table-management-20261006/current/project-evaluation.xlsx",
             "outputs/table-management-20261006/historical-inventory.json"]
    paths += [str(p.relative_to(ROOT)) for p in (ROOT / "artifacts/tokenizer/bitext-bpe-v1").iterdir() if p.is_file()]
    paths += [str(p.relative_to(ROOT)) for p in (ROOT / "src").rglob("*.py")]
    inventory = read("outputs/table-management-20261006/historical-inventory.json")
    assert len(inventory) == 13
    for item in inventory:
        assert digest(item["path"]) == item["sha256"]
        paths.append(item["path"])
    frozen = {Path(p).as_posix(): digest(p) for p in sorted(set(paths))}
    payload = {"experiment_id": plan["experiment_id"], "status": plan["status"],
               "source_sha256": frozen, "ordered_ids": ids,
               "checks": {"pairs": 135, "intents": 27, "historical_tables_unchanged": 13,
                          "baseline_repetition": 63, "baseline_risk": 43},
               "generation_runs": 0, "optimizer_updates": 0, "candidate_implemented": False}
    target = OUT / "freeze.json"
    if target.exists():
        assert json.loads(target.read_text(encoding="utf-8")) == payload, "Frozen evidence changed; do not overwrite"
    else:
        with target.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print("Preregistration verified: 135 IDs, 27 intents; 13 historical tables unchanged; no inference.")


if __name__ == "__main__":
    register()
