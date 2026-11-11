"""Validate saved board readbacks without inferring target/electrical acceptance."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate(lifecycle, fixture, deployment):
    if lifecycle.get('finished') is not True or lifecycle.get('errors'):
        raise ValueError('Camera lifecycle incomplete or failed')
    if (lifecycle.get('physical_uart_enabled') is not False
            or lifecycle.get('motors_enabled') is not False
            or lifecycle.get('control_transport') != 'IN_PROCESS_TEST_NOT_ELECTRICAL_UART'):
        raise ValueError('Unexpected physical execution scope')
    phases = {row['phase']: row for row in lifecycle['phases']}
    control, restart = phases['CONTROL_STOP'], phases['RESTART_SAME_PROCESS']
    expected_modes = ['QR', 'QR', 'QR', 'COLOR', 'RING', 'LANE', 'TOP']
    if [obs['kind'] for obs in control['observations']] != expected_modes:
        raise ValueError('Control sequence or invalid/replayed mode handling failed')
    if control['metrics']['reason'] != 'CONTROL_STOP' or control['closed'] is not True:
        raise ValueError('Control stop did not release the service')
    if control['decoded_frames'] != control['encoded_frames'] or control['rejected_control_frames'] < 2:
        raise ValueError('Control/observation byte checks failed')
    stops = [ack for ack in control['acks'] if ack.get('command_seq') == 8 and ack['status'] == 'STOPPED']
    if len(stops) != 1 or any(ack.get('command_seq') == 60000 for ack in control['acks']):
        raise ValueError('Bad command acknowledged or stop correlation invalid')
    if restart['metrics']['frames'] != 4 or restart['metrics']['reason'] != 'ITERATION_LIMIT':
        raise ValueError('Same-process camera restart incomplete')
    if any(row['metrics']['camera_cleanup_errors'] for row in phases.values()) or lifecycle['capture_cleanup_errors']:
        raise ValueError('Camera cleanup failed')
    if any(obs.get('camera_id') != 2 or obs.get('is_simulated') is not False
           or obs.get('x_mm') is not None or obs.get('target_pose_valid') is not False
           for obs in control['observations']):
        raise ValueError('Capture provenance or calibration boundary violated')
    if (fixture.get('source') != 'FILE_FIXTURE_NOT_CAMERA' or fixture.get('is_simulated') is not True
            or fixture['passed_pixels'] != 10 or fixture['expected_pixels'] != 10
            or fixture['all_mm_denied'] is not True):
        raise ValueError('File-fixture candidate or no-motion validation failed')
    rings = [r['observation'] for r in fixture['results'] if r['observation']['kind'] == 'RING']
    if [r['target_id'] for r in rings] != [1, 2, 3] or not all(r['u_px'] is not None for r in rings):
        raise ValueError('Native ring fixture IDs unresolved')
    if (deployment.get('mismatches') or deployment.get('source_files') != 26
            or deployment['expected'] != deployment['actual']):
        raise ValueError('Deployed source hash mismatch')
    return {'status': 'PASS', 'scope': 'REAL_CSI2_LIFECYCLE_AND_NATIVE_FILE_FIXTURES',
            'native_file_candidates': [10, 10], 'deployed_sources': 26,
            'control_transport': 'IN_PROCESS_TEST_NOT_ELECTRICAL_UART',
            'bad_or_replayed_controls_rejected': control['rejected_control_frames'],
            'physical_uart_validated': False, 'live_target_accuracy_validated': False,
            'live_mm_calibration_validated': False, 'motors_enabled': False}


def main():
    directory = ROOT/'evidence/board'
    def read(name):
        return json.loads((directory/name).read_text(encoding='utf-8'))
    report = validate(read('lifecycle-reconnect.json'), read('native-fixtures-reconnect.json'),
                      read('deployment-reconnect.json'))
    soak, entry = read('capture-soak-reconnect.json'), read('entry-reconnect.json')
    if (soak.get('finished') is not True or soak.get('pass') is not True or soak.get('errors')
            or soak.get('successful_snapshot_count') != 120 or soak.get('unavailable_frames') != 0
            or soak.get('strictly_increasing_ids') is not True
            or soak.get('physical_uart_enabled') is not False or soak.get('motors_enabled') is not False
            or soak['metrics']['camera_cleanup_errors']):
        raise ValueError('Bounded capture did not produce 120 real snapshots cleanly')
    if (entry.get('pass') is not True or entry['entry'] != '/sdcard/transport_mvp/run_board.py'
            or entry['metrics']['frames'] != 3 or entry['metrics']['camera_cleanup_errors']
            or entry['metrics']['motors_enabled'] is not False):
        raise ValueError('Bounded production entry check failed')
    report.update(successful_bounded_snapshots=120, bounded_entry_frames=3)
    (directory/'reconnect-summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
