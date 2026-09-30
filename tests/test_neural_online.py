import json

import pytest

from anima3.neural import live, online


@pytest.mark.parametrize("matches", [1, 2])
def test_cycle_collects_next_batch_with_new_policy(tmp_path, monkeypatch, matches):
    seen = []

    def collect(args, stop):
        assert getattr(args, "known_build", False) == bool(seen)
        assert args.fixture_offset == len(seen) * matches
        seen.append(args.checkpoint)
        return [{"id": str(len(seen))}]

    def update(checkpoint, collection, out, *, seed):
        assert checkpoint == seen[-1]
        return {"checkpoint": str(out)}

    monkeypatch.setattr(live, "run", collect)
    monkeypatch.setattr(online, "update_collection", update)
    assert (
        online.cycle_main(
            [
                "--checkpoint",
                "seed-policy",
                "--user-a",
                "a",
                "--user-b",
                "b",
                "--iterations",
                "2",
                "--matches",
                str(matches),
                "--log-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert str(seen[0]) == "seed-policy"
    assert seen[1] == tmp_path / "round-001" / "candidate"
    result = json.loads((tmp_path / "cycle.json").read_text())
    assert result["status"] == "complete" and len(result["history"]) == 2
    assert result["automatic_promotion"] is False


def test_cycle_records_failure_without_claiming_completion(tmp_path, monkeypatch):
    def collect(args, stop):
        raise RuntimeError("server disconnected")

    monkeypatch.setattr(live, "run", collect)
    with pytest.raises(RuntimeError, match="server disconnected"):
        online.cycle_main(
            [
                "--checkpoint",
                "seed-policy",
                "--user-a",
                "a",
                "--user-b",
                "b",
                "--iterations",
                "2",
                "--log-dir",
                str(tmp_path),
            ]
        )
    result = json.loads((tmp_path / "cycle.json").read_text())
    assert result["status"] == "failed" and result["history"] == []
