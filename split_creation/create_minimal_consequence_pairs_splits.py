import os
import json
import spacy
import argparse

import numpy as np

from string import punctuation
from nltk.corpus import stopwords
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance


def _lemmatize(line, remove_stopwords):
    """ Helper function for obtaining various word representations """

    # strip, replace special tokens
    orig_line = line
    line = line.strip()
    # Remove double space
    line = ' '.join(line.split())
    # Tokenize etc.
    line_nlp = nlp(line)
    spacy_tokens = [elem.text for elem in line_nlp]
    spacy_tokens_lower = [elem.text.lower() for elem in line_nlp]
    spacy_lemmas = list()
    for elem in line_nlp:
        if elem.lemma_ == '-PRON-' or elem.lemma_.isdigit():
            spacy_lemmas.append(elem.lower_)
        else:
            spacy_lemmas.append(elem.lemma_.lower().strip())

    # Generate a mapping between whitespace tokens and SpaCy tokens
    ws_tokens = orig_line.strip().split()
    ws_to_spacy_map = dict()
    spacy_to_ws_map = dict()
    ws_loc = 0
    ws_tok = ws_tokens[ws_loc]

    for spacy_loc, spacy_tok in enumerate(spacy_tokens):
        while True:
            # Map whitespace tokens to be identical to spacy tokens
            if spacy_tok == ws_tok or spacy_tok in ws_tok:
                # Terminate
                if ws_loc >= len(ws_tokens):
                    break

                # Extend maps
                if not ws_to_spacy_map.get(ws_loc, None):
                    ws_to_spacy_map[ws_loc] = list()
                ws_to_spacy_map[ws_loc].append(spacy_loc)
                if not spacy_to_ws_map.get(spacy_loc, None):
                    spacy_to_ws_map[spacy_loc] = list()
                spacy_to_ws_map[spacy_loc].append(ws_loc)

                # Move pointer
                if spacy_tok == ws_tok:
                    ws_loc += 1
                    if ws_loc < len(ws_tokens):
                        ws_tok = ws_tokens[ws_loc]
                else:
                    ws_tok = ws_tok[len(spacy_tok):]
                break
            else:
                ws_loc += 1

    # Assert full coverage of whitespace and SpaCy token sequences by the mapping
    ws_covered = sorted(list(ws_to_spacy_map.keys()))
    spacy_covered = sorted(list(set(list([val for val_list in ws_to_spacy_map.values() for val in val_list]))))
    assert ws_covered == [n for n in range(len(ws_tokens))], \
        'WS-SpaCy mapping does not cover all whitespace tokens: {}; number of tokens: {}'\
        .format(ws_covered, len(ws_tokens))
    assert spacy_covered == [n for n in range(len(spacy_tokens))], \
        'WS-SpaCy mapping does not cover all SpaCy tokens: {}; number of tokens: {}' \
        .format(spacy_covered, len(spacy_tokens))

    if remove_stopwords:
        # Filter out stopwords
        nsw_spacy_lemmas = list()
        for tok_id, tok in enumerate(spacy_tokens_lower):
            if tok not in STOP_WORDS and spacy_lemmas[tok_id] not in STOP_WORDS:
                nsw_spacy_lemmas.append(spacy_lemmas[tok_id])
            else:
                nsw_spacy_lemmas.append('<STPWRD>')

        spacy_lemmas = nsw_spacy_lemmas

    return spacy_lemmas, ws_tokens, spacy_to_ws_map


def split_based_on_consequence_distance(tsv_path, top_list_size, split_size):
    """ Splits collected stories into train / dev / test based on edit distance between consequences. """

    # Relevant fields
    loc_hit_id = 14

    loc_prompt1 = 27
    loc_prompt2 = 28
    loc_prompt3 = 29
    loc_prompt_pick = 40

    loc_situation = 37
    loc_intention = 39
    loc_moral_action = 31
    loc_good_consequence = 35
    loc_immoral_action = 30
    loc_bad_consequence = 34

    # Initialize containers
    stories = dict()
    consequences_edit_distance = dict()
    good_consequence_freqs = dict()
    bad_consequence_freqs = dict()

    # Populate tables
    print('Reading-in AMT results ...')
    with open(tsv_path, 'r', encoding='utf8') as tsv_file:
        for row_id, row in enumerate(tsv_file):
            if row_id == 0:
                continue

            # Read row
            joint_row = row
            row = row.split('\t')
            good_consequence = row[loc_good_consequence].strip()
            good_consequence = good_consequence.translate(str.maketrans(' ', ' ', punctuation))
            bad_consequence = row[loc_bad_consequence].strip()
            bad_consequence = bad_consequence.translate(str.maketrans(' ', ' ', punctuation))

            # Compute edit distance between consequences
            consequences_edit_distance[row_id] = \
                normalized_damerau_levenshtein_distance(good_consequence.split(), bad_consequence.split())

            if top_list_size > 0:
                # Lemmatize consequences and extend tables
                good_consequence_lemmas, good_consequence_tokens, good_spacy_to_ws_map = \
                    _lemmatize(good_consequence, True)
                bad_consequence_lemmas, bad_consequence_tokens, bad_spacy_to_ws_map = \
                    _lemmatize(bad_consequence, True)
                for gl in good_consequence_lemmas:
                    if good_consequence_freqs.get(gl, None) is None:
                        good_consequence_freqs[gl] = 0
                    good_consequence_freqs[gl] += 1
                for bl in bad_consequence_lemmas:
                    if bad_consequence_freqs.get(bl, None) is None:
                        bad_consequence_freqs[bl] = 0
                    bad_consequence_freqs[bl] += 1

                # Store story
                stories[row_id] = [joint_row, row,
                                   good_consequence_lemmas, bad_consequence_lemmas,
                                   good_consequence_tokens, bad_consequence_tokens,
                                   good_spacy_to_ws_map, bad_spacy_to_ws_map]
            else:
                # Store story
                stories[row_id] = [joint_row, row]

    # Sort stories by the edit distance between the consequences
    sorted_dist_diff_tpls = sorted([(story_id, dist) for story_id, dist in consequences_edit_distance.items()],
                                   reverse=False, key=lambda x: x[1])

    # Detect bias terms
    top_biased_lemmas_dict = dict()
    if top_list_size > 0:
        print('Isolating biased terms ...')
        term_freq_diffs = dict()
        for lem in good_consequence_freqs.keys():
            if lem == '<STPWRD>':
                continue
            if bad_consequence_freqs.get(lem, None) is None:
                term_freq_diffs[lem] = good_consequence_freqs[lem]
            else:
                term_freq_diffs[lem] = good_consequence_freqs[lem] - bad_consequence_freqs[lem]
        for lem in bad_consequence_freqs.keys():
            if good_consequence_freqs.get(lem, None) is None:
                term_freq_diffs[lem] = -bad_consequence_freqs[lem]

        # Sort by frequency difference
        freq_diff_tpls = sorted([(lem, freq, np.abs(freq)) for lem, freq in term_freq_diffs.items()],
                                reverse=True, key=lambda x: x[2])
        top_biased_lemma_tpls = freq_diff_tpls[:top_list_size]
        top_biased_lemmas_dict = {tpl[0]: tpl[1] for tpl in top_biased_lemma_tpls}

        for k, v, v2 in top_biased_lemma_tpls:
            print(k, v)

    # Build splits
    print('-' * 20)
    print('Creating splits ...')

    # Select stories for test and dev splits, first
    sorted_story_ids, sorted_consequence_distances = zip(*sorted_dist_diff_tpls)
    # Keep track of consequence distance
    train_distances = list()
    dev_distances = list()
    test_distances = list()
    if top_list_size > 0:
        test_story_ids = list()
        dev_story_ids = list()
        train_story_ids = list()
        for iid, story_id in enumerate(sorted_story_ids):
            # Check which lemmas differ between consequences
            _, row, good_lemmas, bad_lemmas, good_tokens, bad_tokens, good_map, bad_map = stories[story_id]
            unique_good_lemmas = list(set(good_lemmas) - set(bad_lemmas))
            unique_bad_lemmas = list(set(bad_lemmas) - set(good_lemmas))
            biased_good_lemmas = [lem for lem in unique_good_lemmas if top_biased_lemmas_dict.get(lem, 0) > 0]
            biased_bad_lemmas = [lem for lem in unique_bad_lemmas if top_biased_lemmas_dict.get(lem, 0) < 0]

            # If the contrasting lemmas are morally charged, place story in test, otherwise place it in dev / train
            is_in_test = False
            if len(biased_good_lemmas) == 0 and len(biased_bad_lemmas) == 0:
                if len(test_story_ids) < split_size:
                    test_story_ids.append(story_id)
                    test_distances.append(sorted_consequence_distances[iid])
                    is_in_test = True
            if not is_in_test:
                if len(dev_story_ids) < split_size:
                    dev_story_ids.append(story_id)
                    dev_distances.append(sorted_consequence_distances[iid])
                else:
                    train_story_ids.append(story_id)
                    train_distances.append(sorted_consequence_distances[iid])
    else:
        test_story_ids = sorted_story_ids[: split_size]
        test_distances = sorted_consequence_distances[: split_size]

        dev_story_ids = sorted_story_ids[split_size: split_size * 2]
        dev_distances = sorted_consequence_distances[split_size: split_size * 2]

        train_story_ids = sorted_story_ids[split_size * 2:]
        train_distances = sorted_consequence_distances[split_size * 2:]

    # Write to files
    for task in ['consequence+action_cls', 'consequence+action+context_cls',
                 'consequence|action_gen', 'consequence|action+context_gen']:
        tsv_dir = '/'.join(tsv_path.split('/')[:-1])
        task_dir = os.path.join(tsv_dir, task)
        challenge_dir = os.path.join(task_dir, 'lexical_bias')
        if not os.path.exists(task_dir):
            os.mkdir(task_dir)
        if not os.path.exists(challenge_dir):
            os.mkdir(challenge_dir)

        train_path = os.path.join(challenge_dir, 'train.jsonl')
        dev_path = os.path.join(challenge_dir, 'dev.jsonl')
        test_path = os.path.join(challenge_dir, 'test.jsonl')

        for story_ids, path in [(train_story_ids, train_path),
                                (dev_story_ids, dev_path),
                                (test_story_ids, test_path)]:
            lines = list()
            # Parse story
            for s_id in story_ids:
                story = stories[s_id][1]
                # Parse row
                if '1' in story[loc_prompt_pick].strip():
                    norm = story[loc_prompt1].strip()
                elif '2' in story[loc_prompt_pick].strip():
                    norm = story[loc_prompt2].strip()
                else:
                    norm = story[loc_prompt3].strip()

                hit_id = story[loc_hit_id].strip()
                situation = story[loc_situation].strip()
                intention = story[loc_intention].strip()
                moral_action = story[loc_moral_action].strip()
                good_consequence = story[loc_good_consequence].strip()
                immoral_action = story[loc_immoral_action].strip()
                bad_consequence = story[loc_bad_consequence].strip()

                # Construct split entry
                line1 = {'ID': hit_id + '1',
                         'norm': norm,
                         'situation': situation,
                         'intention': intention,
                         'moral_action': moral_action,
                         'good_consequence': good_consequence,
                         'label': '1'}
                line2 = {'ID': hit_id + '2',
                         'norm': norm,
                         'situation': situation,
                         'intention': intention,
                         'moral_action': moral_action,
                         'bad_consequence': bad_consequence,
                         'label': '0'}
                line3 = {'ID': hit_id + '3',
                         'norm': norm,
                         'situation': situation,
                         'intention': intention,
                         'immoral_action': immoral_action,
                         'bad_consequence': bad_consequence,
                         'label': '1'}
                line4 = {'ID': hit_id + '4',
                         'norm': norm,
                         'situation': situation,
                         'intention': intention,
                         'immoral_action': immoral_action,
                         'good_consequence': good_consequence,
                         'label': '0'}

                if task in ['consequence+action_cls', 'consequence|action_gen']:
                    for line in [line1, line2, line3, line4]:
                        del line['norm']
                        del line['situation']
                        del line['intention']
                if task in ['consequence|action_gen', 'consequence|action+context_gen']:
                    lines += [line1, line3]
                else:
                    lines += [line1, line2, line3, line4]

            # Dump to file
            with open(path, 'w', encoding='utf8') as out_f:
                for line_id, line in enumerate(lines):
                    if line_id > 0:
                        out_f.write('\n')
                    out_f.write(json.dumps(line))

            if path.endswith('train.jsonl'):
                print('{:s} TRAIN size: {:d}'.format(task, len(lines)))
                print('Mean TRAIN degree of isolation: {:.2f}'.format(np.mean(train_distances)))
                print('Saved to {:s}'.format(train_path))
                print('-' * 5)
            if path.endswith('dev.jsonl'):
                print('{:s} DEV size: {:d}'.format(task, len(lines)))
                print('Mean DEV degree of isolation: {:.2f}'.format(np.mean(dev_distances)))
                print('Saved to {:s}'.format(dev_path))
                print('-' * 5)
            if path.endswith('test.jsonl'):
                print('{:s} TEST size: {:d}'.format(task, len(lines)))
                print('Mean TEST degree of isolation: {:.2f}'.format(np.mean(test_distances)))
                print('Saved to {:s}'.format(test_path))
                print('-' * 5)


if __name__ == '__main__':
    # Instantiate processing pipeline
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'textcat'])
    # Import stopword list
    STOP_WORDS = [w for w in stopwords.words('english') if w != 'not']

    parser = argparse.ArgumentParser()
    parser.add_argument('--stories_path', type=str, required=True,
                        help='path to file containing the collected stories')
    parser.add_argument('--top_list_size',
                        type=int, default=0, help='number of most frequent biased terms to consider')
    parser.add_argument('--split_size', type=int, default=1000, help='size of dev / test splits')
    args = parser.parse_args()

    split_based_on_consequence_distance(args.stories_path, args.top_list_size, args.split_size)


