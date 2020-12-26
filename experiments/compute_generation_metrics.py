import json
import argparse

from utils import compute_gen_metrics


def compute_gen_metrics_wrapper(output_file_path):
    """ Computes evaluation metrics for generative models directly on model output. """

    def _read_jsonl(input_file):
        """ Reads a .jsonl file. """
        records = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                records.append(json.loads(line))
        return records

    # Read-in results file
    file_records = _read_jsonl(output_file_path)
    inputs = list()
    preds = list()
    targets = list()

    for frec in file_records:
        inputs.append(frec['prefix'])
        preds.append(frec['prediction'])
        targets.append(frec['target'])

    # Compute metrics
    gen_metrics = compute_gen_metrics(preds, targets)

    # Report
    print('***** Test results *****')
    for key in sorted(gen_metrics.keys()):
        print('  %s = %s', key, str(gen_metrics[key]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_output_path', type=str, required=True,
                        help='path to file containing model predictions to be evaluated')
    args = parser.parse_args()
    compute_gen_metrics_wrapper(args.model_output_path)
