"""Run RSPTrack on a benchmark (LaSOT-ext by default)."""
import argparse
import _init_paths
from lib.project import DEFAULT_CONFIG, DEFAULT_DATASET, canonical_experiment

def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name=DEFAULT_DATASET,
                sequence=None, debug=0, threads=1, num_gpus=1, checkpoint=None):
    from lib.test.evaluation import get_dataset, Tracker
    from lib.test.evaluation.running import run_dataset
    tracker_name, tracker_param = canonical_experiment(tracker_name, tracker_param)
    dataset = get_dataset(dataset_name)
    if sequence is not None:
        dataset = [dataset[sequence]]
    tracker = Tracker(tracker_name, tracker_param, dataset_name, run_id)
    tracker.checkpoint_override = checkpoint
    run_dataset(dataset, [tracker], debug, threads, num_gpus=num_gpus)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tracker_name', nargs='?', default='rsptrack')
    parser.add_argument('tracker_param', nargs='?', default=DEFAULT_CONFIG)
    parser.add_argument('--runid', type=int)
    parser.add_argument('--dataset_name', default=DEFAULT_DATASET,
                        help='lasot, lasot_extension_subset, tnl2k, otb99_lang')
    parser.add_argument('--sequence', help='Sequence name or zero-based index')
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--checkpoint', help='Optional explicit checkpoint path')
    args = parser.parse_args()
    sequence = int(args.sequence) if args.sequence and args.sequence.isdigit() else args.sequence
    run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name,
                sequence, args.debug, args.threads, args.num_gpus, args.checkpoint)

if __name__ == '__main__':
    main()
