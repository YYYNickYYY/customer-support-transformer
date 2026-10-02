import pytest

from my_project_1 import split_data


def test_template_normalization():
    assert split_data.template_key(" WHERE is {{Order Number}}? \n") == "where is ?"
    assert split_data.template_key("Hi {{Name}}   there") == "hi there"
    assert split_data.template_key("{{Name}}") == ""


def test_group_isolation_conservation_and_reproducibility():
    rows = [{"instruction": f"Question {i} {{{{Name}}}}", "intent": "a"} for i in range(20)]
    rows += [{"instruction": " QUESTION 0 {{Other}} ", "intent": "b"}]
    splits = split_data.split_records(rows)
    assert splits == split_data.split_records(rows)
    groups = [{r["template_id"] for r in part} for part in splits.values()]
    assert [len(g) for g in groups] == [16, 2, 2]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert sorted(r["source_row"] for part in splits.values() for r in part) == list(range(21))
    assert len({r["id"] for part in splits.values() for r in part}) == 21


def test_reject_wrong_source_before_output_creation(tmp_path):
    source = tmp_path / "wrong.csv"
    source.write_text("instruction\nx\n")
    with pytest.raises(ValueError, match="SHA-256"):
        split_data.build_split(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_existing_frozen_output_is_preserved(tmp_path, monkeypatch):
    source = tmp_path / "sample.csv"
    source.write_text("instruction\n" + "\n".join(f"q{i}" for i in range(20)))
    monkeypatch.setattr(split_data, "SOURCE_SHA256", split_data.hashlib.sha256(source.read_bytes()).hexdigest())
    output = tmp_path / "out"
    split_data.build_split(source, output)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        split_data.build_split(source, output, seed=123)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
