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
DEPLOY_WORDING = re.compile(r"deploy", re.IGNORECASE)
NOT_DEPLOYED = re.compile(r"\bnot deployed\b", re.IGNORECASE)
# The only fields an index entry may have; an unknown key is refused, not ignored.
EXPERIMENT_KEYS = {
    "id", "title", "dates", "source_ref", "domain", "hypothesis", "configuration", "outcome",
    "result", "files", "context_files", "candidates", "server_changes", "parent_evidence",
    "deployment", "deployment_evidence", "recorded_at_utc",
}  # fmt: skip
CANDIDATE_KEYS = {"label", "sha256", "selection", "deployment", "deployment_evidence"}

# Reviewed labels, frozen. Per experiment: result, (SHA-256 prefix, selection) per candidate,
# a digest of the narrative fields and a digest of every candidate's label and note.
# Changing any of them needs a deliberate update here.
EXPECTED = {
    "E01-ten-duels": ("negative", (), "90215ef46461cd07", "4f53cda18c2baa0c"),
    "E02-neural-starter": (
        "diagnostic",
        (("7c0cbe1a", "not_gated"), (None, "rejected"), ("2953462a", "not_gated")),
        "0d1155a76bebb631",
        "ce894e3b24fdc3d1",
    ),
    "E03-neural-continuation": (
        "simulator_only",
        (
            ("22bc333e", "offline_selected"),
            ("46fb2317", "rejected"),
            ("558a7dde", "rejected"),
            ("fe5796b4", "not_gated"),
        ),
        "cd3ab1fc5d85da99",
        "770425ae208ea2dd",
    ),
    "E04-hundred-live-duels": (
        "negative",
        (("4fd33a57", "not_gated"), ("ee89aab5", "not_gated")),
        "dbd25a79b9919301",
        "6a815ec7a6852aad",
    ),
    "E05-melee-harm-probe": ("diagnostic", (), "70ef33e3b59d3ff2", "4f53cda18c2baa0c"),
    "E06-cast-dump-timing": ("diagnostic", (), "719ee1ff47422ee7", "4f53cda18c2baa0c"),
    "E07-curriculum-league-v2": (
        "simulator_only",
        (("39abd306", "registry_selected"), ("e5915b8d", "registry_selected"), (None, "rejected")),
        "edbe4b392b1afc25",
        "28191fab8c75731c",
    ),
    "E08-live-pilot-28": (
        "failed_gate",
        (("ec9b2a7c", "rejected"), ("e5915b8d", "registry_retained")),
        "5ca0375b6a18ca64",
        "15004864edb654a0",
    ),
    "E09-skill-clock-simulator": (
        "simulator_only",
        (("0c6ca921", "registry_selected"), (None, "rejected")),
        "d1a40717cfa14658",
        "b57bc5ddf852215f",
    ),
    "E10-skill-clock-live-imitation": (
        "invalidated",
        (("03b06cc7", "invalidated"), ("d0096ef9", "invalidated")),
        "a0b979f7d66b9f80",
        "331e13d29f099ca9",
    ),
    "E11-supply-correction": ("diagnostic", (), "47d1b53513b9d866", "4f53cda18c2baa0c"),
    "E12-visibility-diagnosis": ("diagnostic", (), "83e48ae109787c7b", "4f53cda18c2baa0c"),
    "E13-supply-fixed-trial": (
        "failed_gate",
        (("5e56ab68", "rejected"), ("3d0f7b45", "rejected")),
        "eb50aa045d6ea32b",
        "3df416bc7c42faa7",
    ),
    "E14-v8-bc-screen": (
        "failed_gate",
        (("5e56ab68", "rejected"), ("0c6ca921", "registry_retained")),
        "31bd54f965c7f814",
        "10609731db7f4860",
    ),
    "E15-threat-memory": (
        "failed_gate",
        (("772c0fc6", "registry_selected"), (None, "rejected")),
        "75a04c492866062e",
        "ac2ef1ed5c34b89f",
    ),
    "E16-bc-rehearsal": (
        "failed_gate",
        (("772c0fc6", "registry_selected"), (None, "rejected")),
        "f01e1a9cfb730243",
        "fa838132538db8f2",
    ),
    "E17-elapsed-time-discount": (
        "failed_gate",
        (("772c0fc6", "registry_selected"), (None, "rejected")),
        "1c40c922dc7e7e5c",
        "742ec61710d7214d",
    ),
    "E18-time-rehearsal": (
        "failed_gate",
        (("772c0fc6", "registry_selected"), (None, "rejected")),
        "bafdc712cbd99942",
        "968e69734352a534",
    ),
    "E19-incumbent-continuation": (
        "failed_gate",
        ((None, "rejected"), ("0c6ca921", "registry_retained")),
        "efe329715e6702c4",
        "ca7023ce8790a30e",
    ),
    "E20-recovered-bc-200": (
        "failed_gate",
        (("5e03b624", "rejected"), ("0c6ca921", "registry_retained")),
        "72b64664f74b5cfb",
        "d81b5fe5c0d26f07",
    ),
    "E21-recovered-bc-replication-400": (
        "failed_gate",
        (("5e03b624", "rejected"), ("0c6ca921", "registry_retained")),
        "9913b02fec42fec0",
        "d81b5fe5c0d26f07",
    ),
    "E22-reference-bc-fit": (
        "diagnostic",
        (("0b3fb0a1", "not_gated"), ("af3cb4ad", "not_gated")),
        "a7e93c4faa552ad1",
        "7d0324f8f706c0c5",
    ),
    "E23-reference-simulator": (
        "simulator_only",
        (("0b3fb0a1", "rejected"), ("af3cb4ad", "offline_selected")),
        "9518a86c96802327",
        "5ba78e90a7d48b57",
    ),
    "E24-reference-bc-live-200": (
        "passed_gate",
        (("af3cb4ad", "registry_selected"), ("0c6ca921", "rollback")),
        "f6fc5323fb89d423",
        "07b79dc71dfa20b4",
    ),
    "E25-jev-shadow": ("negative", (), "c86836fa231778f7", "4f53cda18c2baa0c"),
    "E26-tactical-pilot": ("diagnostic", (), "04f90409f3da9978", "4f53cda18c2baa0c"),
    "E27-potion-defense": ("diagnostic", (), "9e2a6319cc4bb24a", "4f53cda18c2baa0c"),
}
NARRATIVE = (
    "title",
    "dates",
    "source_ref",
    "domain",
    "hypothesis",
    "configuration",
    "outcome",
    "server_changes",
    "parent_evidence",
)
# A registry selection that only a Markdown ledger records (league-v2.md), not a JSON registry.
TEXT_ONLY_SELECTIONS = {
    (
        "E07-curriculum-league-v2",
        "39abd306f8b6dc69a10c9e26c58cd4617a4c2e67d4f8b1715291b378a8c40796",
    ),
}
NOT_IMPORTED_DIGEST = "7e78fa87a8ac4e8898a8359e95bfc71516ef224c58ddf1cc44a1236167e12fc9"
DEPLOYMENT_DIGEST = "461fe27be4f0c172165fdbf38fd6718c42af76eef81c7909782175fff594675c"
VOCABULARY_DIGEST = "020848037c0967aee4a0b6354728df7a96d14fef339a7b02f7f373b6a6528dcb"
PARENT_SHA256 = {
    "P1-reference-bc-audit": "087248ab359d38b4f743db9ff381e4d9b06eb8e41beb6782b728a79258493c4f",
    "P2-model-evidence-doc": "e867fadabe597aa619359d1d22b1c0a6fc297a70f964ebbd113c47a616339109",
    "P3-readiness-smoke": "e8653850d36de956f9b6dd843bb52013dd42365a0cab4318ce00e477e8af5134",
    "P4-model-readiness-doc": "dc3cccff9974904e0f46d439f2710d0b350b13a496c4d3e381edb199a6d68de1",
    "P5-deployment-map": "9cc6a970f4f005ffed0ffc5b41ae2d55b7c7665344b5866bfc0a64c2fe1ce082",
}


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
    subjects += [
        (f"{e['id']}:{c['label']}", c) for e in index["experiments"] for c in e["candidates"]
    ]
    candidates = [id(c) for e in index["experiments"] for c in e["candidates"]]
    for e in index["experiments"]:
        if set(e) != EXPERIMENT_KEYS:
            raise ValueError(
                f"{e['id']}: unknown or missing keys {sorted(set(e) ^ EXPERIMENT_KEYS)}"
            )
        for c in e["candidates"]:
            if set(c) not in (CANDIDATE_KEYS, CANDIDATE_KEYS | {"note"}):
                raise ValueError(f"{e['id']}:{c.get('label')}: unknown or missing candidate keys")
    for name, subject in subjects:
        status = subject["deployment"]
        if status not in allowed:
            raise ValueError(f"{name}: unknown deployment status {status!r}")
        if status == "not_deployed":
            if subject["deployment_evidence"] is not None:
                raise ValueError(f"{name}: evidence given for an undeployed subject")
            # A candidate's free text may say "not deployed", never anything else about
            # deployment. Every text field is checked except the status itself.
            # (Experiment narratives mention shard deployments and are frozen by digest.)
            texts = subject.items() if id(subject) in candidates else ()
            for field, text in texts:
                if field == "deployment" or not isinstance(text, str):
                    continue
                if DEPLOY_WORDING.search(NOT_DEPLOYED.sub("", text)):
                    raise ValueError(f"{name}: {field} claims deployment: {text!r}")
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
    for c in [
        tip,
        lineage["common_ancestor"],
        *extra,
        *(c["parent"] for c in lineage["extra_commits"]),
    ]:
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
    assert "17" in data[tip["bytes"] :].decode("utf-8")


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
        text = "".join(
            (REPO / p).read_text(encoding="utf-8") for p in e["files"] + e["context_files"]
        )
        for c in e["candidates"]:
            if c["sha256"]:
                assert c["sha256"] in text, (e["id"], c["label"])


def test_gate_decisions_match_the_imported_records():
    found = []
    for path in json_records():
        for pointer, gate in gate_objects(read_json(path)):
            found.append((path, pointer, gate["passed"], gate["reason"], gate.get("domain")))
    indexed = [
        (g["file"], g["pointer"], g["passed"], g["reason"], g["domain"])
        for g in INDEX["gate_records"]
    ]
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
                c["selection"] == "rejected" for c in e["candidates"]
            ), e["id"]
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
            deployment="public_deployment", deployment_evidence=i["experiments"][23]["files"][1]
        ),
        lambda i: i["experiments"][0].update(deployment="deployed"),
        lambda i: i["experiments"][21]["candidates"][0].update(note="deployed publicly"),
        lambda i: i["experiments"][21]["candidates"][1].update(label="reference BC (deployed)"),
        lambda i: i["experiments"][23]["candidates"][0].update(
            note="selected in the experimental registry; not deployed, later deployed"
        ),
        lambda i: i["experiments"][6]["candidates"][1].update(note="Deployment to the arena"),
        lambda i: i["experiments"][21]["candidates"][0].update(comment="deployed publicly"),
        lambda i: i["experiments"][21]["candidates"][0].update(comment="harmless"),
        lambda i: i["experiments"][21]["candidates"][0].pop("deployment_evidence"),
        lambda i: i["experiments"][21].update(comment="deployed publicly"),
        lambda i: i["experiments"][21]["candidates"][0].update(selection="deployed_publicly"),
    ):
        index = copy.deepcopy(INDEX)
        mutate(index)
        with pytest.raises(ValueError):
            check_deployment_claims(index)


def test_candidate_text_is_frozen_and_refuses_deployment_wording():
    """A note such as E22's set to "deployed publicly" fails both checks."""
    for field, value in (("note", "deployed publicly"), ("label", "plain BC, renamed")):
        index = copy.deepcopy(INDEX)
        e22 = next(e for e in index["experiments"] if e["id"] == "E22-reference-bc-fit")
        e22["candidates"][0][field] = value
        with pytest.raises(AssertionError, match="E22-reference-bc-fit"):
            check_reviewed(index)
        if field == "note":
            with pytest.raises(ValueError, match="note claims deployment"):
                check_deployment_claims(index)
    # An unknown key is refused by both checks, whatever it says.
    for target in ("candidate", "experiment"):
        index = copy.deepcopy(INDEX)
        e22 = next(e for e in index["experiments"] if e["id"] == "E22-reference-bc-fit")
        (e22["candidates"][0] if target == "candidate" else e22)["comment"] = "harmless"
        with pytest.raises(AssertionError, match="E22-reference-bc-fit"):
            check_reviewed(index)
        with pytest.raises(ValueError, match="unknown or missing"):
            check_deployment_claims(index)
    e24 = next(e for e in INDEX["experiments"] if e["id"] == "E24-reference-bc-live-200")
    assert "not deployed" in e24["candidates"][0]["note"]  # the one allowed phrase


def test_readme_experiment_rows_list_every_identified_candidate():
    rows = {}
    for line in (REPO / "docs/history/README.md").read_text().splitlines():
        if re.match(r"\| E\d\d \|", line):
            rows[line.split("|")[1].strip()] = line.split("|")[5]
    assert sorted(rows) == sorted(e["id"][:3] for e in INDEX["experiments"])
    for e in INDEX["experiments"]:
        row = rows[e["id"][:3]]
        for c in e["candidates"]:
            if c["sha256"]:
                assert f"`{c['sha256'][:8]}`" in row, (e["id"], c["label"])
            assert c["selection"] in row, (e["id"], c["label"], c["selection"])


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
    recorded = {
        (x["manifest"], x["location"].rsplit("/", 1)[1], x["bytes"], x["sha256"]) for x in blobs
    }
    assert recorded == described
    for x in INDEX["not_imported"]:
        assert x["reason"].strip()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def candidate_text(experiment):
    return [[c["label"], c.get("note")] for c in experiment["candidates"]]


def check_reviewed(index):
    """Raise AssertionError if any reviewed label, narrative or candidate text changed."""
    assert [e["id"] for e in index["experiments"]] == list(EXPECTED)
    for e in index["experiments"]:
        result, candidates, narrative, text = EXPECTED[e["id"]]
        assert set(e) == EXPERIMENT_KEYS, e["id"]
        for c in e["candidates"]:
            assert set(c) in (CANDIDATE_KEYS, CANDIDATE_KEYS | {"note"}), e["id"]
        assert e["result"] == result, e["id"]
        got = tuple(
            (c["sha256"][:8] if c["sha256"] else None, c["selection"]) for c in e["candidates"]
        )
        assert got == candidates, e["id"]
        assert digest([e[k] for k in NARRATIVE])[:16] == narrative, e["id"]
        assert digest(candidate_text(e))[:16] == text, e["id"]
    assert digest(index["vocabulary"]) == VOCABULARY_DIGEST


def test_reviewed_labels_and_narratives_are_frozen():
    check_reviewed(INDEX)


def registry_objects(experiment):
    for path in experiment["files"]:
        if not path.endswith(".json"):
            continue
        stack = [read_json(path)]
        while stack:
            obj = stack.pop()
            if isinstance(obj, dict):
                if HEX64.fullmatch(str(obj.get("selected_sha", ""))):
                    yield path, obj
                stack.extend(obj.values())
            elif isinstance(obj, list):
                stack.extend(obj)


def derived_registry_labels(experiment):
    """Registry labels that follow from the experiment's own JSON records."""
    labels = set()
    for path, reg in registry_objects(experiment):
        if "gate" in reg:  # live registry: a passing gate selects, a failing one retains
            passed = reg["gate"]["passed"]
            labels.add(
                (reg["selected_sha"], "registry_selected" if passed else "registry_retained")
            )
            if passed and reg.get("rollback_checkpoint") == reg.get("incumbent"):
                labels.add((read_json(path)["models"]["incumbent"]["policy_sha"], "rollback"))
        else:  # simulator run registry: its warmstart is new only if the BC gate passed
            passed = reg["bc_gate"]["passed"]
            labels.add(
                (reg["selected_sha"], "registry_selected" if passed else "registry_retained")
            )
    return labels


def test_registry_labels_are_derived_from_the_records():
    for e in INDEX["experiments"]:
        claimed = {
            (c["sha256"], c["selection"])
            for c in e["candidates"]
            if c["selection"] in ("registry_selected", "registry_retained", "rollback")
        }
        text_only = {
            (sha, "registry_selected") for eid, sha in TEXT_ONLY_SELECTIONS if eid == e["id"]
        }
        assert claimed == derived_registry_labels(e) | text_only, e["id"]
        selected = {sha for sha, label in derived_registry_labels(e) if label != "rollback"}
        for c in e["candidates"]:
            if c["sha256"] in selected:
                assert c["selection"] in ("registry_selected", "registry_retained"), e["id"]


def test_results_follow_the_gate_records():
    by_experiment = {}
    for g in INDEX["gate_records"]:
        by_experiment.setdefault(g["experiment"], []).append(g)
    for e in INDEX["experiments"]:
        gates = by_experiment.get(e["id"], [])
        live = [g for g in gates if g["domain"] == "servuo"]
        sim = [g for g in gates if g["domain"] == "approximate-pre-aos-simulator"]
        assert len(live) + len(sim) == len(gates), e["id"]
        if e["result"] == "passed_gate":
            assert live and all(g["passed"] for g in live), e["id"]
        elif e["result"] == "failed_gate":
            assert gates and not any(g["passed"] for g in live), e["id"]
            assert any(not g["passed"] for g in gates), e["id"]
        elif e["result"] == "simulator_only":
            assert not live, e["id"]
            assert not sim or any(g["passed"] for g in sim), e["id"]
        else:
            assert not live, e["id"]


def test_not_imported_and_parent_evidence_are_pinned():
    rows = sorted((x["kind"], x["location"], x.get("sha256")) for x in INDEX["not_imported"])
    assert digest(rows) == NOT_IMPORTED_DIGEST
    for x in INDEX["not_imported"]:
        assert x["reason"].strip()
        assert not x["location"].startswith(("/Users", "/home", "~"))
    pluto = [x for x in INDEX["not_imported"] if "pluto" in x["location"].lower()]
    assert len(pluto) == 7
    assert {p["id"]: p["sha256"] for p in INDEX["parent_evidence"]} == PARENT_SHA256
    for p in INDEX["parent_evidence"]:
        assert p["git_blob"] and HEX40.fullmatch(p["git_blob"])
    assert INDEX["parent_evidence"][4]["asserts"] == {
        "arena_worker_or_learner_unit_observed": False
    }


def test_deployment_findings_are_stated_and_pinned():
    findings = INDEX["deployment_findings"]
    assert findings["basis"] and all(b.strip() for b in findings["basis"])
    assert findings["scope"] and findings["limits"]
    assert findings["scope"][0] == "No neural policy was publicly deployed."
    not_imported = {x["location"] for x in INDEX["not_imported"]}
    assert set(findings["scope_sources"]) <= not_imported
    assert digest(findings) == DEPLOYMENT_DIGEST


def commit_available(commit):
    if shutil.which("git") is None:
        return False
    done = subprocess.run(
        ["git", "-C", str(REPO), "cat-file", "-e", f"{commit}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    return done.returncode == 0


SOURCE_COMMITS = sorted({f["source_commit"] for f in INDEX["files"]})


@pytest.mark.parametrize("commit", SOURCE_COMMITS)
def test_recorded_blobs_exist_in_the_lineage_commits(commit):
    if not commit_available(commit):
        pytest.skip(f"{commit[:7]} is not in this clone; digests are checked above")

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False
        )

    for f in INDEX["files"]:
        if f["source_commit"] != commit:
            continue
        out = git("rev-parse", f"{commit}:{f['source_path']}")
        assert out.stdout.strip() == f["git_blob"], f["path"]
        for key in ("introduced_in", "last_changed_in"):
            assert git("merge-base", "--is-ancestor", f[key], commit).returncode == 0
        tip = f.get("training_tip_version")
        if tip and commit_available(tip["commit"]):
            out = git("rev-parse", f"{tip['commit']}:{f['source_path']}")
            assert out.stdout.strip() == tip["git_blob"]
    for x in INDEX["not_imported"]:
        if "git_blob" in x and x["location"].startswith(commit + ":"):
            assert git("rev-parse", x["location"]).stdout.strip() == x["git_blob"], x["location"]


def test_extra_commits_name_a_public_ref():
    for c in INDEX["lineage"]["extra_commits"]:
        assert c["public_ref"].startswith("archive/")
