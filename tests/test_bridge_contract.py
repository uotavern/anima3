"""Offline bridge regressions; mocked subprocesses never connect to a server."""
import io
import json
from unittest.mock import Mock

import pytest

from anima3 import body
from anima3.contract import SCHEMA_VERSION, Observation, Player


def process(*responses):
    return Mock(stdin=io.StringIO(), stderr=io.StringIO(),
                stdout=io.StringIO(''.join(json.dumps(r) + '\n' for r in responses)))


def bridge(*responses):
    return body.BridgeBody(process(*responses), {'schema_version': SCHEMA_VERSION}, 12)


def test_pump_failure_is_not_reported_as_zero_packets():
    client = bridge({'ok': False, 'error': 'transport stopped',
                     'diagnostics': {'bytes_in': 123, 'packets_in': 7}})
    with pytest.raises(body.BodyError, match='transport stopped'):
        client.pump(10)
    assert client.diagnostics == {'bytes_in': 123, 'packets_in': 7}


def test_successful_legacy_bridge_needs_no_diagnostics():
    client = bridge({'ok': True, 'applied': 0}, {'ok': True, 'obs': {'player': {}}})
    assert client.pump(0) == 0
    assert client.observe().player == Player()
    assert client.diagnostics == {}


def test_optional_diagnostics_keep_last_snapshot_across_old_responses():
    client = bridge({'ok': True, 'applied': 2, 'diagnostics': {'bytes_in': 10}},
                    {'ok': True, 'applied': 0},
                    {'ok': True, 'applied': 1, 'diagnostics': None},
                    {'ok': True, 'applied': 3, 'diagnostics': {'bytes_in': 20}})
    assert client.pump(10) == 2
    assert client.diagnostics == {'bytes_in': 10}
    client.pump(0)
    client.pump(0)
    assert client.diagnostics == {'bytes_in': 10}
    client.pump(10)
    assert client.diagnostics == {'bytes_in': 20}


def test_non_object_reply_fails_at_bridge_boundary():
    with pytest.raises(body.BodyError, match='JSON object'):
        bridge([]).pump(0)


@pytest.mark.parametrize('monitor,explicit,expected', [
    (None, False, 'headless'), (8801, False, 'spectator'), (8801, True, 'override'),
])
def test_main_headless_monitor_selection_is_preserved(monkeypatch, tmp_path,
                                                     monitor, explicit, expected):
    binaries = {name: tmp_path / name for name in ['headless', 'spectator', 'override']}
    for binary in binaries.values():
        binary.touch()
    monkeypatch.setattr(body, 'DEFAULT_BRIDGE', binaries['headless'])
    monkeypatch.setattr(body, 'DEFAULT_MONITOR_BRIDGE', binaries['spectator'])
    popen = Mock(return_value=process({'event': 'ready', 'schema_version': SCHEMA_VERSION}))
    monkeypatch.setattr(body.subprocess, 'Popen', popen)
    body.BridgeBody.spawn('example.invalid', 2594, 'test', 'test-only', monitor_port=monitor,
                         binary=binaries['override'] if explicit else None)
    assert popen.call_args.args[0][0] == str(binaries[expected])
    if monitor is not None:
        assert popen.call_args.kwargs['env']['ANIMA_MONITOR_PORT'] == str(monitor)


def test_schema_version_mismatch_is_still_rejected(monkeypatch, tmp_path):
    binary = tmp_path / 'bridge'
    binary.touch()
    proc = process({'event': 'ready', 'schema_version': SCHEMA_VERSION + 1})
    monkeypatch.setattr(body.subprocess, 'Popen', Mock(return_value=proc))
    with pytest.raises(body.BodyError, match='unsupported bridge schema'):
        body.BridgeBody.spawn('example.invalid', 2594, 'test', 'test-only', binary=binary)
    proc.kill.assert_called_once()


@pytest.mark.parametrize('pending', [None, False, True])
def test_legacy_target_presence_has_no_invented_cursor(pending):
    obs = Observation.from_json({'pending_target': pending})
    assert obs.pending_target is bool(pending)
    assert (obs.target_cursor_id, obs.target_cursor_flag, obs.target_cursor_type) == (None,) * 3


def test_zero_cursor_fields_are_preserved_for_held_target_tracking():
    obs = Observation.from_json({'pending_target': {
        'cursor_id': 0, 'cursor_flag': 0, 'target_type': 0,
    }})
    assert obs.pending_target is True
    assert (obs.target_cursor_id, obs.target_cursor_flag, obs.target_cursor_type) == (0, 0, 0)
    changed = Observation.from_json({'pending_target': {'cursor_id': 44, 'target_type': 1}})
    assert changed.target_cursor_id == 44 and changed.target_cursor_type == 1
    assert changed.target_cursor_flag is None


def test_effects_are_optional_and_preserved_without_private_inference():
    effect = {'kind': 0, 'src': 57, 'tgt': 58, 'graphic': 0x36D4, 'unknown': 'future'}
    payload = {'recent_effects': [effect]}
    obs = Observation.from_json(payload)
    assert obs.recent_effects == [effect]
    payload['recent_effects'].clear()
    assert obs.recent_effects == [effect]
    assert Observation.from_json({}).recent_effects == []
    assert Observation.from_json({'recent_effects': None}).recent_effects == []


def test_additive_fields_do_not_shift_old_positional_arguments():
    obs = Observation(Player(), [], [], [], True, True)
    assert obs.pending_target is True and obs.war is True
    assert obs.target_cursor_id is None and obs.recent_effects == []
