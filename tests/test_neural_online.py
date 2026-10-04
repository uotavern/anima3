import json

import pytest

from anima3.neural import live, online
from anima3.sparring import TrainingIdleTimeout


def test_rejected_completed_match_counts_without_updating_policy(tmp_path, monkeypatch):
    seen = []

    def collect(args, stop):
        seen.append(str(args.checkpoint))
        return [{"match_id": str(len(seen))}]

    def reject(*args, **kwargs):
        raise online.NoEligibleEpisodes([{"reasons": ["deadline"]}])

    monkeypatch.setattr(live, "run", collect)
    monkeypatch.setattr(online, "update_collection", reject)
    assert online.cycle_main([
        "--checkpoint", "seed", "--user-a", "a", "--user-b", "b",
        "--iterations", "2", "--matches", "1", "--log-dir", str(tmp_path),
    ]) == 0
    result = json.loads((tmp_path / 'cycle.json').read_text())
    assert result['status'] == 'complete'
    assert seen == ['seed', 'seed']
    assert len(result['history']) == 2
    assert all(h['learning']['skipped'] for h in result['history'])


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


@pytest.mark.parametrize("recover", [True, False])
def test_idle_retry_is_bounded_and_keeps_fixture(tmp_path, monkeypatch, recover):
    attempts = []

    def collect(args, stop):
        attempts.append((args.fixture_offset, str(args.checkpoint)))
        args.log_dir.mkdir(parents=True, exist_ok=True)
        (args.log_dir / "diagnostic.json").write_text('{}')
        if not recover or len(attempts) == 1:
            raise TrainingIdleTimeout("no fresh state")
        return [{"match_id": "only-one-match"}]

    monkeypatch.setattr(live, "run", collect)
    monkeypatch.setattr(online, "update_collection", lambda *a, **kw: {"updated": True})
    monkeypatch.setattr(online.threading.Event, "wait", lambda *a: False)
    argv = ["--checkpoint", "seed", "--user-a", "a", "--user-b", "b",
            "--iterations", "1", "--matches", "1", "--log-dir", str(tmp_path)]
    if recover:
        assert online.cycle_main(argv) == 0
    else:
        with pytest.raises(TrainingIdleTimeout):
            online.cycle_main(argv)
    result = json.loads((tmp_path / "cycle.json").read_text())
    assert len(attempts) == (2 if recover else 3)
    assert set(attempts) == {(0, "seed")}
    assert len(result['history']) == (1 if recover else 0)
    assert (tmp_path / 'round-001/idle-retry-1/diagnostic.json').exists()
