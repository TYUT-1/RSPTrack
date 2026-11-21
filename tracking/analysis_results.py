"""Evaluate saved RSPTrack predictions using AUC, P and normalized P."""
import argparse
import _init_paths
from lib.project import DEFAULT_CONFIG, DEFAULT_DATASET, canonical_experiment

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracker_name', default='rsptrack')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--dataset_name', default=DEFAULT_DATASET)
    parser.add_argument('--runid', type=int)
    parser.add_argument('--plot', action='store_true', help='Also save success/precision plots')
    args = parser.parse_args()
    from lib.test.analysis.plot_results import print_results, plot_results
    from lib.test.evaluation import get_dataset, trackerlist
    name, config = canonical_experiment(args.tracker_name, args.config)
    trackers = trackerlist(name=name, parameter_name=config, dataset_name=args.dataset_name,
                           run_ids=args.runid, display_name='RSPTrack' if config == DEFAULT_CONFIG else config)
    dataset = get_dataset(args.dataset_name)
    report = args.dataset_name + '/' + config
    options = dict(merge_results=True, plot_types=('success', 'prec', 'norm_prec'), force_evaluation=True)
    print_results(trackers, dataset, report, seq_eval=False, **options)
    if args.plot:
        plot_results(trackers, dataset, report, **options)

if __name__ == '__main__':
    main()
