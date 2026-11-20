"""Offline checks of the imported training-lineage experiment records (unit 9)."""

import copy
import hashlib
import json
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
INDEX_PATH = REPO / "docs" / "history" / "index.json"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


def load_index():
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


INDEX = load_index()
FILES = {f["path"]: f for f in INDEX["files"]}
ROOT = INDEX["lineage"]["import_root"]


def git_blob(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def read_json(path):
    return json.loads((REPO / path).read_bytes())


def resolve(doc, pointer):
    for part in pointer.split("/")[1:]:
        part = part.replace("~1", "/").replace("~0", "~")
        doc = doc[int(part)] if isinstance(doc, list) else doc[part]
    return doc


def gate_objects(obj, pointer=""):
    if isinstance(obj, dict):
        if isinstance(obj.get("passed"), bool) and isinstance(obj.get("reason"), str):
            yield pointer, obj
        for key, value in obj.items():
            yield from gate_objects(value, f"{pointer}/{key.replace('~', '~0').replace('/', '~1')}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from gate_objects(value, f"{pointer}/{i}")


def values_of(obj, key):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                yield v
            yield from values_of(v, key)
    elif isinstance(obj, list):
        for v in obj:
            yield from values_of(v, key)


def json_records():
    return [f["path"] for f in INDEX["files"] if f["path"].endswith(".json")]


def check_deployment_claims(index):
    """Refuse any public-deployment claim that lacks a linked deployment evidence record."""
    allowed = set(index["vocabulary"]["deployment"])
    evidence = {f["path"] for f in index["files"] if f["kind"] == "deployment_evidence"}
    subjects = [(e["id"], e) for e in index["experiments"]]
    subjects += [(f"{e['id']}:{c['label']}", c) for e in index["experiments"]
                 for c in e["candidates"]]
    for name, subject in subjects:
        status = subject["deployment"]
        if status not in allowed:
            raise ValueError(f"{name}: unknown deployment status {status!r}")
        if status == "not_deployed":
            if subject["deployment_evidence"] is not None:
                raise ValueError(f"{name}: evidence given for an undeployed subject")
        elif subject["deployment_evidence"] not in evidence:
            raise ValueError(f"{name}: {status} without a linked deployment evidence record")


def test_every_indexed_file_matches_its_recorded_digest():
    for path, f in FILES.items():
        data = (REPO / path).read_bytes()
        assert len(data) == f["bytes"], path
        assert hashlib.sha256(data).hexdigest() == f["sha256"], path
        assert git_blob(data) == f["git_blob"], path
        assert path == f"{ROOT}/{f['source_path']}"


def test_import_root_holds_exactly_the_indexed_files():
    on_disk = {p.relative_to(REPO).as_posix() for p in (REPO / ROOT).rglob("*") if p.is_file()}
    assert on_disk == set(FILES)
    assert len(FILES) == 77
    assert not [p for p in on_disk if p.endswith((".pt", ".bin", ".zip"))]


def test_source_commits_are_full_hashes_of_the_recorded_lineage():
    lineage = INDEX["lineage"]
    tip = lineage["training_tip"]["commit"]
    extra = {c["commit"] for c in lineage["extra_commits"]}
    for c in [tip, lineage["common_ancestor"], *extra, *(c["parent"] for c in lineage["extra_commits"])]:
        assert HEX40.fullmatch(c)
    for f in INDEX["files"]:
        for key in ("source_commit", "introduced_in", "last_changed_in"):
            assert HEX40.fullmatch(f[key]), (f["path"], key)
        assert f["source_commit"] in {tip, *extra}
        assert f["revisions"] >= 1
        if f["revisions"] == 1:
            assert f["introduced_in"] == f["last_changed_in"]
    from_extra = {p for c in lineage["extra_commits"] for p in c["files"]}
    assert from_extra == {p for p, f in FILES.items() if f["source_commit"] in extra}


def test_hundred_duel_record_extends_the_training_tip_version():
    [path] = INDEX["lineage"]["extra_commits"][0]["files"]
    f = FILES[path]
    tip = f["training_tip_version"]
    assert f["source_ref"] == "feat/arena-100-duels"
    assert tip["commit"] == INDEX["lineage"]["training_tip"]["commit"]
    data = (REPO / path).read_bytes()
    prefix = data[: tip["bytes"]]
    assert len(data) > tip["bytes"]
    assert hashlib.sha256(prefix).hexdigest() == tip["sha256"]
    assert git_blob(prefix) == tip["git_blob"]
    assert "17" in data[tip["bytes"]:].decode("utf-8")


def test_experiments_are_in_chronological_order():
    times = []
    for e in INDEX["experiments"]:
        first = min(FILES[p]["introduced_at_utc"] for p in e["files"])
        assert e["recorded_at_utc"] == first, e["id"]
        times.append(first)
    assert times == sorted(times)
    for f in INDEX["files"]:
        local = datetime.fromisoformat(f["introduced_at"]).astimezone(UTC)
        assert local.strftime("%Y-%m-%dT%H:%M:%SZ") == f["introduced_at_utc"], f["path"]
    ids = [e["id"] for e in INDEX["experiments"]]
    assert ids == sorted(ids) and len(ids) == len(set(ids)) == 27


def test_every_file_belongs_to_exactly_one_experiment():
    primary = [p for e in INDEX["experiments"] for p in e["files"]]
    assert sorted(primary) == sorted(FILES)
    assert len(primary) == len(set(primary))
    for e in INDEX["experiments"]:
        assert set(e["context_files"]) <= set(FILES), e["id"]
        assert not set(e["context_files"]) & set(e["files"]), e["id"]


def test_experiment_fields_use_the_status_vocabulary():
    vocab = INDEX["vocabulary"]
    for e in INDEX["experiments"]:
        for key in ("title", "dates", "hypothesis", "configuration", "outcome", "source_ref"):
            assert e[key].strip(), (e["id"], key)
        assert e["result"] in vocab["result"], e["id"]
        assert set(e["domain"]) <= set(vocab["domain"]), e["id"]
        for c in e["candidates"]:
            assert c["selection"] in vocab["selection"], (e["id"], c["label"])
            assert c["sha256"] is None or HEX64.fullmatch(c["sha256"])


def test_candidate_identities_come_from_the_experiment_records():
    for e in INDEX["experiments"]:
        text = "".join((REPO / p).read_text(encoding="utf-8")
                       for p in e["files"] + e["context_files"])
        for c in e["candidates"]:
            if c["sha256"]:
                assert c["sha256"] in text, (e["id"], c["label"])


def test_gate_decisions_match_the_imported_records():
    found = []
    for path in json_records():
        for pointer, gate in gate_objects(read_json(path)):
            found.append((path, pointer, gate["passed"], gate["reason"], gate.get("domain")))
    indexed = [(g["file"], g["pointer"], g["passed"], g["reason"], g["domain"])
               for g in INDEX["gate_records"]]
    assert sorted(found) == sorted(indexed)
    assert len(found) == 48
    assert sum(not g[2] for g in found) == 36
    owners = {p: e["id"] for e in INDEX["experiments"] for p in e["files"]}
    for g in INDEX["gate_records"]:
        assert g["experiment"] == owners[g["file"]]


def test_failed_and_rejected_candidates_are_preserved():
    experiments = INDEX["experiments"]
    rejected = [c for e in experiments for c in e["candidates"] if c["selection"] == "rejected"]
    assert len(rejected) == 17
    gated_fail = {g["experiment"] for g in INDEX["gate_records"] if not g["passed"]}
    for e in experiments:
        if e["result"] == "failed_gate":
            assert e["id"] in gated_fail or any(
                c["selection"] == "rejected" for c in e["candidates"]), e["id"]
    results = [e["result"] for e in experiments]
    assert results.count("passed_gate") == 1
    assert results.count("failed_gate") == 10
    passed_live = [g for g in INDEX["gate_records"] if g["passed"] and g["domain"] == "servuo"]
    assert {g["experiment"] for g in passed_live} == {"E24-reference-bc-live-200"}


def test_no_public_deployment_without_linked_evidence():
    check_deployment_claims(INDEX)
    assert not [f for f in INDEX["files"] if f["kind"] == "deployment_evidence"]
    assert INDEX["deployment_findings"]["public_model_deployment_found"] is False
    for e in INDEX["experiments"]:
        assert e["deployment"] == "not_deployed", e["id"]
        for c in e["candidates"]:
            assert c["deployment"] == "not_deployed", (e["id"], c["label"])


def test_deployment_check_refuses_claims_without_evidence():
    for mutate in (
        lambda i: i["experiments"][23].update(deployment="public_deployment"),
        lambda i: i["experiments"][23]["candidates"][0].update(deployment="public_deployment"),
        lambda i: i["experiments"][23]["candidates"][0].update(
            deployment="public_deployment",
            deployment_evidence=i["experiments"][23]["files"][1]),
        lambda i: i["experiments"][0].update(deployment="deployed"),
    ):
        index = copy.deepcopy(INDEX)
        mutate(index)
        with pytest.raises(ValueError):
            check_deployment_claims(index)


def test_source_records_never_claim_public_deployment():
    flags = [v for path in json_records() for v in values_of(read_json(path), "public_deployment")]
    assert flags and all(v is False for v in flags)


def test_selected_reference_bc_agrees_with_registry_and_parent_evidence():
    e24 = next(e for e in INDEX["experiments"] if e["id"] == "E24-reference-bc-live-200")
    final = read_json(next(p for p in e24["files"] if p.endswith("reference-live-final.json")))
    selected = next(c for c in e24["candidates"] if c["selection"] == "registry_selected")
    registry = final["registry"]
    assert registry["selected_sha"] == selected["sha256"]
    assert registry["gate"]["passed"] is True
    assert registry["public_deployment"] is False
    parent = {p["id"]: p for p in INDEX["parent_evidence"]}
    assert set(e24["parent_evidence"]) <= set(parent)
    audit = parent["P1-reference-bc-audit"]["asserts"]
    smoke = parent["P3-readiness-smoke"]["asserts"]
    assert audit["selected_policy_sha256"] == selected["sha256"] == smoke["candidate_sha256"]
    assert audit["registry_sha256"] == final["artifact_sha256"]["registry.json"]
    assert smoke["historical_registry_sha256_after"] == audit["registry_sha256"]
    assert audit["recorded_public_deployment"] is False and smoke["public_deployment"] is False
    assert smoke["training_updates"] == 0
    assert parent["P5-deployment-map"]["asserts"]["arena_worker_or_learner_unit_observed"] is False
    for p in INDEX["parent_evidence"]:
        assert HEX40.fullmatch(p["commit"]) and HEX64.fullmatch(p["sha256"])


def test_local_registry_hashes_corroborate_the_records():
    registries = [x for x in INDEX["not_imported"] if x["kind"] == "local_registry"]
    assert len(registries) == 13
    matched = [x for x in registries if "matches_record" in x]
    assert len(matched) == 4
    for x in matched:
        ref = x["matches_record"]
        assert resolve(read_json(ref["file"]), ref["pointer"]) == x["sha256"]


def test_checkpoint_blobs_are_recorded_by_their_manifests():
    blobs = [x for x in INDEX["not_imported"] if x["kind"] == "checkpoint_blob"]
    assert len(blobs) == 12
    manifests = [p for p, f in FILES.items() if f["kind"] == "checkpoint_manifest"]
    described = set()
    for path in manifests:
        doc = read_json(path)
        for part in ("weights", "optimizer", "rng"):
            d = doc[part]
            described.add((path, d["file"], d["bytes"], d["sha256"]))
    recorded = {(x["manifest"], x["location"].rsplit("/", 1)[1], x["bytes"], x["sha256"])
                for x in blobs}
    assert recorded == described
    for x in INDEX["not_imported"]:
        assert x["reason"].strip()


def lineage_objects_available():
    if shutil.which("git") is None:
        return False
    tip = INDEX["lineage"]["training_tip"]["commit"]
    done = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{tip}^{{commit}}"],
                          capture_output=True, check=False)
    return done.returncode == 0


@pytest.mark.skipif(not lineage_objects_available(),
                    reason="lineage commits are not in this clone; digests are checked above")
def test_recorded_blobs_exist_in_the_lineage_commits():
    def git(*args):
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              check=False)

    for f in INDEX["files"]:
        out = git("rev-parse", f"{f['source_commit']}:{f['source_path']}")
        assert out.stdout.strip() == f["git_blob"], f["path"]
        for key in ("introduced_in", "last_changed_in"):
            assert git("merge-base", "--is-ancestor", f[key], f["source_commit"]).returncode == 0
        if "training_tip_version" in f:
            tip = f["training_tip_version"]
            out = git("rev-parse", f"{tip['commit']}:{f['source_path']}")
            assert out.stdout.strip() == tip["git_blob"]
    for x in INDEX["not_imported"]:
        if "git_blob" in x:
            out = git("rev-parse", x["location"])
            assert out.stdout.strip() == x["git_blob"], x["location"]
