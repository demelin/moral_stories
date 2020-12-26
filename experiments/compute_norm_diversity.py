# From https://stackoverflow.com/questions/32441605/
# generating-ngrams-unigrams-bigrams-etc-from-a-large-corpus-of-txt-files-and-t

import json
import spacy
import string
import argparse
from nltk.util import ngrams
from collections import Counter


def _preprocess(line):

    """ Helper function for obtaining various word representations """
    # Strip punctuation, lower
    line = line.translate(translator).lower().strip()
    # Lemmatize
    line_nlp = nlp(line)
    spacy_lemmas = list()
    for elem in line_nlp:
        if elem.lemma_ == '-PRON-' or elem.lemma_.isdigit():
            spacy_lemmas.append(elem.lower_)
        else:
            spacy_lemmas.append(elem.lemma_.lower().strip())

    return spacy_lemmas


def compute_distinct_ngrams(output_file_path, max_ngram):
    """ Finds and compares distinct (lemma-)ngram ratios between model outputs and corresponding gold references """

    def _read_jsonl(input_file):
        """ Reads a .jsonl file. """
        records = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                records.append(json.loads(line))
        return records

    # Read-in results file
    file_records = _read_jsonl(output_file_path)
    preds = list()
    targets = list()

    for frec in file_records:
        # Remove punctuation and lemmatize strings
        preds.append(_preprocess(frec['prediction']))
        targets.append(_preprocess(frec['target']))

    # Find and count ngrams
    preds_ngrams = {n + 1: list() for n in range(max_ngram)}
    preds_ngrams[1] = [tok for sent in preds for tok in sent]
    for pred_tokens in preds:
        for n in range(1, max_ngram+1):
            preds_ngrams[n] += ngrams(pred_tokens, n)
    for n in preds_ngrams:
        preds_ngrams[n] = Counter(preds_ngrams[n])

    target_ngrams = {n + 1: list() for n in range(max_ngram)}
    target_ngrams[1] = [tok for sent in targets for tok in sent]
    for target_tokens in targets:
        for n in range(1, max_ngram+1):
            target_ngrams[n] += ngrams(target_tokens, n)
    for n in target_ngrams:
        target_ngrams[n] = Counter(target_ngrams[n])

    # Compute score
    pred_score = \
        sum([len(preds_ngrams[n]) / sum(preds_ngrams[n].values()) for n in preds_ngrams.keys()]) / max_ngram
    target_score = \
        sum([len(target_ngrams[n]) / sum(target_ngrams[n].values()) for n in target_ngrams.keys()]) / max_ngram

    print('***** Diversity scores: *****')
    print('Predictions score {:.4f}'.format(pred_score))
    print('Target score {:.4f}'.format(target_score))
    print('Prediction-to-target ratio: {:.4f}'.format(pred_score / target_score))


if __name__ == '__main__':

    # Instantiate processing pipeline
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'textcat'])
    pct_stripper = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_output_path', type=str, required=True,
                        help='path to file containing model predictions to be evaluated')
    parser.add_argument('--max_ngram_size', type=int, default=4,
                        help='longest n-gram to consider for the diversity metric computation')
    args = parser.parse_args()

    compute_distinct_ngrams(args.model_output_path, args.max_ngram_size)
