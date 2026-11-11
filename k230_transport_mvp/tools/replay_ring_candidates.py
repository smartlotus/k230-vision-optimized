"""Host replay of saved CanMV file candidates; this never contacts a board."""
import argparse
import hashlib
import json
from pathlib import Path
from vision.ring import identify_ring

ROOT = Path(__file__).resolve().parents[1]


def replay(source, layout):
    source, layout = Path(source), Path(layout)
    original = json.loads(source.read_text(encoding='utf-8'))
    reference = json.loads(layout.read_text(encoding='utf-8'))
    if original.get('is_simulated') is not True or reference.get('is_simulated') is not True:
        raise ValueError('This replay requires explicitly simulated file fixtures')
    gates = {int(k): v for k, v in reference['center_gates'].items()}
    observations = [item['observation'] for item in original['results']
                    if item['observation']['kind'] == 'RING']
    if len(observations) != 3:
        raise ValueError('Expected three saved ring observations')
    results = []
    for target_id, observation in zip((1, 2, 3), observations):
        groups = observation.get('ring_candidates', [])
        candidate, reason = identify_ring(groups, target_id, {'verified': True, 'center_gates': gates})
        results.append({'target_id': target_id, 'old_status': observation['status'],
                        'old_reason': observation.get('reason'), 'candidate_groups': len(groups),
                        'result': 'PIXEL_CANDIDATE' if candidate else 'ID_UNKNOWN',
                        'reason': reason, 'candidate': candidate, 'x_mm': None, 'y_mm': None,
                        'is_simulated': True, 'target_pose_valid': False})
    return {'scope': 'HOST_REPLAY_OF_SAVED_BOARD_FILE_CANDIDATES',
            'board_rerun': False, 'live_verified': False,
            'reference_source': reference['source'],
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'layout_sha256': hashlib.sha256(layout.read_bytes()).hexdigest(),
            'results': results, 'resolved': sum(r['candidate'] is not None for r in results),
            'denominator': 3, 'motion_enabled': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'evidence/board/native-fixtures-attempt02.json')
    parser.add_argument('--layout', type=Path, default=ROOT/'data/board_fixtures/synthetic-ring-layout.json')
    parser.add_argument('--output', type=Path, default=ROOT/'evidence/ring-candidate-replay.json')
    args = parser.parse_args()
    report = replay(args.input, args.layout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('scope', 'resolved', 'denominator', 'board_rerun', 'motion_enabled')}))
    return 0 if report['resolved'] == report['denominator'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
