import os
import json
import argparse

import numpy as np

from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer


def embed_norms(norm_list):
    """ Generates sentence-level embeddings of moral norms. """
    print('Computing norm embeddings ...')
    norm_embeddings = embedder.encode(norm_list)
    return norm_embeddings


def split_based_on_norm(tsv_path, n_clusters, split_size):
    """ Splits collected stories into train / dev / test based on moral norm (dis-)similarity. """

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

    # Read in stories
    stories = dict()
    print('Reading-in AMT results ...')
    with open(tsv_path, 'r', encoding='utf8') as tsv_file:
        for row_id, row in enumerate(tsv_file):
            if row_id == 0:
                continue
            else:
                stories[row_id] = row.split('\t')

    # Isolate norms per story
    norm_to_story = dict()
    for story_id in stories.keys():
        story = stories[story_id]
        # Identify norm
        prompt1 = story[loc_prompt1].strip()
        prompt2 = story[loc_prompt2].strip()
        prompt3 = story[loc_prompt3].strip()
        prompt_pick = story[loc_prompt_pick].strip()
        # Identify norm
        if '1' in prompt_pick:
            norm = prompt1
        elif '2' in prompt_pick:
            norm = prompt2
        else:
            norm = prompt3

        if norm in norm_to_story.keys():
            print(norm)

        norm_to_story[norm] = stories[story_id]

    # Embed norms
    norm_list = list(norm_to_story.keys())
    norm_embeddings = embed_norms(norm_list)

    # Report
    print('# of unique norms in corpus: {:d}'.format(len(norm_list)))
    print('# of obtained norm embeddings: {:d}'.format(len(norm_embeddings)))

    # Cluster embeddings
    print('Clustering norm embeddings ...')
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters)
    clustering_model.fit(norm_embeddings)
    cluster_assignment = clustering_model.labels_
    clustered_norms = {cluster_id: list() for cluster_id in range(n_clusters)}
    for norm_id, cluster_id in enumerate(cluster_assignment):
        clustered_norms[cluster_id].append((norm_list[norm_id], norm_embeddings[norm_id]))

    # Compute cluster centroids
    print('Sorting clusters ...')
    centroids = dict()
    for cluster_id in clustered_norms.keys():
        cluster_centroid = np.mean([tpl[1] for tpl in clustered_norms[cluster_id]], axis=0).reshape(1, -1)
        centroids[cluster_id] = cluster_centroid

    # Sort clusters by their "degree of isolation" (mean minimal distance to other clusters)
    isolation_scores = dict()
    for cluster_id1 in centroids.keys():
        centroid_distances = list()
        for cluster_id2 in centroids.keys():
            if cluster_id1 == cluster_id2:
                continue
            centroid_distances.append(cdist(centroids[cluster_id1], centroids[cluster_id2], metric='cosine'))
        isolation_scores[cluster_id1] = min(centroid_distances)
    sorted_clusters = sorted([(c, i) for c, i in isolation_scores.items()], reverse=True, key=lambda x: x[1])

    # Construct splits
    print('Creating splits ...')
    train_norms = list()
    dev_norms = list()
    test_norms = list()
    train_isolation_scores = list()
    dev_isolation_scores = list()
    test_isolation_scores = list()

    for cluster_id, dist in sorted_clusters:
        if len(test_norms) < split_size:
            test_norms += [tpl[0] for tpl in clustered_norms[cluster_id]]
            test_isolation_scores += [dist] * len(clustered_norms[cluster_id])
        elif len(dev_norms) < split_size:
            dev_norms += [tpl[0] for tpl in clustered_norms[cluster_id]]
            dev_isolation_scores += [dist] * len(clustered_norms[cluster_id])
        else:
            train_norms += [tpl[0] for tpl in clustered_norms[cluster_id]]
            train_isolation_scores += [dist] * len(clustered_norms[cluster_id])
    # Trim sets
    train_norms = test_norms[split_size:] + train_norms
    train_isolation_scores = test_isolation_scores[split_size:] + train_isolation_scores
    test_norms = test_norms[:split_size]
    test_isolation_scores = test_isolation_scores[:split_size]

    train_norms = dev_norms[split_size:] + train_norms
    train_isolation_scores = dev_isolation_scores[split_size:] + train_isolation_scores
    dev_norms = dev_norms[:split_size]
    dev_isolation_scores = dev_isolation_scores[:split_size]

    # Write to files
    print('=' * 20)

    # Write to files
    for task in ['action_cls', 'action+norm_cls', 'action+context_cls', 'action+context+consequence_cls',
                 'action|context_gen', 'action|context+consequence_gen',
                 'norm|actions_gen', 'norm|actions+context_gen', 'norm|actions+context+consequences_gen',
                 'consequence+action_cls', 'consequence+action+context_cls',
                 'consequence|action_gen', 'consequence|action+context_gen']:
        tsv_dir = '/'.join(tsv_path.split('/')[:-1])
        task_dir = os.path.join(tsv_dir, task)
        challenge_dir = os.path.join(task_dir, 'norm_overlap')
        if not os.path.exists(task_dir):
            os.mkdir(task_dir)
        if not os.path.exists(challenge_dir):
            os.mkdir(challenge_dir)

        train_path = os.path.join(challenge_dir, 'train.jsonl')
        dev_path = os.path.join(challenge_dir, 'dev.jsonl')
        test_path = os.path.join(challenge_dir, 'test.jsonl')

        for norms, path in [(train_norms, train_path),
                            (dev_norms, dev_path),
                            (test_norms, test_path)]:
            lines = list()
            # Parse story
            for norm in norms:
                story = norm_to_story[norm]
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

                # Construct split entry, depending on the task
                if task in ['action_cls', 'action+norm_cls', 'action+context_cls', 'action+context+consequence_cls',
                            'action|context_gen', 'action|context+consequence_gen',
                            'norm|actions_gen', 'norm|actions+context_gen', 'norm|actions+context+consequences_gen']:
                    line1 = {'qID': hit_id + '1',
                             'norm': norm,
                             'situation': situation,
                             'intention': intention,
                             'moral_action': moral_action,
                             'label': '1'}
                    line2 = {'qID': hit_id + '0',
                             'norm': norm,
                             'situation': situation,
                             'intention': intention,
                             'immoral_action': immoral_action,
                             'label': '0'}

                    if task in ['action_cls', 'action+norm_cls', 'norm|actions_gen']:
                        del line1['situation']
                        del line1['intention']
                        del line2['situation']
                        del line2['intention']
                    if task in ['action_cls']:
                        del line1['norm']
                        del line2['norm']

                    if task in ['action+context+consequence_cls', 'action|context+consequence_gen']:
                        line1['good_consequence'] = good_consequence
                        line2['bad_consequence'] = bad_consequence

                    if task in ['norm|actions_gen', 'norm|actions+context_gen',
                                'norm|actions+context+consequences_gen']:
                        line1['immoral_action'] = immoral_action
                        if task in ['norm|actions+context+consequences_gen']:
                            line1['good_consequence'] = good_consequence
                            line1['bad_consequence'] = bad_consequence
                        lines += [line1]
                    else:
                        lines += [line1, line2]

                else:
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
                print('Mean TRAIN degree of isolation: {:.2f}'.format(np.mean(train_isolation_scores)))
                print('Saved to {:s}'.format(train_path))
                print('-' * 5)
            if path.endswith('dev.jsonl'):
                print('{:s} DEV size: {:d}'.format(task, len(lines)))
                print('Mean DEV degree of isolation: {:.2f}'.format(np.mean(dev_isolation_scores)))
                print('Saved to {:s}'.format(dev_path))
                print('-' * 5)
            if path.endswith('test.jsonl'):
                print('{:s} TEST size: {:d}'.format(task, len(lines)))
                print('Mean TEST degree of isolation: {:.2f}'.format(np.mean(test_isolation_scores)))
                print('Saved to {:s}'.format(test_path))
                print('-' * 5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stories_path', type=str, required=True,
                        help='path to file containing the collected stories')
    parser.add_argument('--cluster_size', type=int, default=1000, help='size of norm clusters')
    parser.add_argument('--split_size', type=int, default=1000, help='size of dev / test splits')
    args = parser.parse_args()

    # Initialize embedder
    embedder = SentenceTransformer('bert-base-nli-mean-tokens')

    split_based_on_norm(args.stories_path, args.cluster_size, args.split_size)
