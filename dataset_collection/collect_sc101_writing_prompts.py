import csv
import json
import string
import pickle
import random
import argparse

import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def construct_json_from_sc(sc_path_tsv):
    """ Reads-in SocialChem and converts it to a situation -> norms dictionary. """

    # Relevant SocialChem fields
    source_id = 0
    split_id = 2
    agr_id = 3
    rot_cat_id = 4
    rot_mf_id = 5
    jud_id = 11
    sit_id = 17
    rot_id = 19

    # Initialize table
    sc_table = dict()

    # Read in file
    print('Reading-in SC database ...')
    with open(sc_path_tsv, 'r', encoding='utf8') as tsv_file:
        csv_reader = csv.reader(tsv_file, delimiter='\t')
        for row_id, row in enumerate(csv_reader):
            if row_id == 0:
                continue
            else:
                split = row[split_id].strip()
                if split != 'train':
                    continue

                # Parse row
                source = row[source_id].strip()
                rot_cat = row[rot_cat_id].strip()
                rot_mf = row[rot_mf_id].strip()
                jud = row[jud_id]
                agr = row[agr_id]
                sit = row[sit_id].strip()
                rot = row[rot_id].strip()

                if len(jud) == 0 or len(agr) == 0:
                    continue

                # Extend table
                if sc_table.get(rot, None) is None:
                    sc_table[rot] = [rot_cat, rot_mf, source, int(jud), int(agr), sit, row_id]

            if row_id > 0 and row_id % 10000 == 0:
                print('Read {:d} lines'.format(row_id))

    # Report number of unique RoTs
    print('Found {:d} unique RoTs'.format(len(sc_table.keys())))

    # Filter out unsuitable RoTs
    filtered_sc_table = dict()
    for rot in sc_table.keys():
        # RoT should belong to ethics / norms
        if 'morality-ethics' not in sc_table[rot][0] and 'social-norms' not in sc_table[rot][0]:
            continue
        # RoT should not be controversial
        if sc_table[rot][4] < 3:
            continue
        # RoT should not be neutral
        if sc_table[rot][3] == 0:
            continue
        # RoTs should not be too long (as shorter RoTs are expected to be more general)
        if len(rot.strip().translate(str.maketrans('', '', string.punctuation)).split()) > 10:
            continue
        # RoTs shouldn't contain commas
        if ',' in rot:
            continue
        filtered_sc_table[rot] = sc_table[rot]

    sc_table = filtered_sc_table
    all_rots = list(sc_table.keys())
    print('Retained {:d} unique RoTs after filtering'.format(len(all_rots)))

    # Save to JSON
    print('Saving database to JSON ...')
    sc_path_json = sc_path_tsv[:-3] + 'json'
    with open(sc_path_json, 'w', encoding='utf8') as jsp:
        json.dump(sc_table, jsp, indent=3, sort_keys=True, ensure_ascii=False)
    return sc_path_json


def embed_norms(sc_path_json):
    """ Generates sentence-level embeddings of moral norms. """

    print('Computing RoT embeddings ...')

    # Read-in JSON tables
    with open(sc_path_json, 'r', encoding='utf8') as sc_obj:
        sc_table = json.load(sc_obj)

    # Get norms as list
    rot_list = list(sc_table.keys())
    # Get embeddings
    rot_embeddings = embedder.encode(rot_list)
    # Build embedding dictionary
    embed_dict = {rot_list[i]: rot_embeddings[i] for i in range(len(rot_list))}

    # Pickle embeddings
    sc_path_pkl = sc_path_json[:-4] + 'embeddings.pkl'
    with open(sc_path_pkl, 'wb') as pickle_file:
        pickle.dump(embed_dict, pickle_file)
    return sc_path_pkl


def cluster_norms(sc_embeddings_path, n_clusters):
    """ Clusters semantically similar social norms together and samples one norm per cluster. """

    print('Clustering RoT embeddings ...')

    # Unpickle embeddings
    with open(sc_embeddings_path, 'rb') as pickle_file:
        embed_dict = pickle.load(pickle_file)
    rot_list = list()
    rot_embeddings = list()
    for rot, embed in embed_dict.items():
        rot_list.append(rot)
        rot_embeddings.append(embed)

    # Cluster embeddings
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters)
    clustering_model.fit(rot_embeddings)
    cluster_assignment = clustering_model.labels_
    clustered_rots = {cluster_id: list() for cluster_id in range(n_clusters)}
    for rot_id, cluster_id in enumerate(cluster_assignment):
        clustered_rots[cluster_id].append((rot_list[rot_id], rot_embeddings[rot_id]))

    # For each cluster, return RoT closest to centroid
    centroid_rots = dict()
    for cluster_id in clustered_rots.keys():
        # Compute centroid
        cluster_centroid = np.mean([tpl[1] for tpl in clustered_rots[cluster_id]], axis=0).reshape(1, -1)
        # Find most-central RoT
        central_rot = None
        max_rot_sim = -1
        for tpl in clustered_rots[cluster_id]:
            if central_rot is None:
                # Initial assignment
                central_rot = tpl[0]
            else:
                # Update assignment
                rot_sim = cosine_similarity(tpl[1].reshape(1, -1), cluster_centroid)
                if rot_sim > max_rot_sim:
                    max_rot_sim = rot_sim
                    central_rot = tpl[0]
        centroid_rots[cluster_id] = (central_rot, [tpl[0] for tpl in clustered_rots[cluster_id]])

    # Pickle clustering results
    sc_path_pkl = sc_embeddings_path[:-3] + 'clustered.pkl'
    with open(sc_path_pkl, 'wb') as pickle_file:
        pickle.dump(clustered_rots, pickle_file)

    # Dump centroid RoTs to JSON
    print('Saving centroid RoTs to JSON ...')
    centroid_rots_path_json = sc_embeddings_path[:-14] + 'centroid_rots.json'
    with open(centroid_rots_path_json, 'w', encoding='utf8') as crp:
        json.dump(centroid_rots, crp, indent=3, sort_keys=True, ensure_ascii=False)

    # Write to .csv file
    sampled_rots = list(set([v[0] for v in centroid_rots.values()]))
    random.shuffle(sampled_rots)
    # Make triples
    prompt_triples = list()
    curr_triple = list()
    for sr in sampled_rots:
        curr_triple.append(sr)
        if len(curr_triple) == 3:
            prompt_triples.append(curr_triple)
            curr_triple = list()
    lines_written = 0
    centroid_rots_path_csv = sc_embeddings_path[:-14] + 'centroid_rots.csv'
    with open(centroid_rots_path_csv, 'w') as in_file:
        in_file.write('prompt_norm_1,prompt_norm_2,prompt_norm_3\n')
        for pt in prompt_triples:
            in_file.write(','.join(pt)+'\n')
            lines_written += 1
    print('Wrote {:d} lines to the .csv file'.format(lines_written))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sc101_path', type=str, required=True,
                        help='path to file containing the Social-Chemistry-101 file')
    parser.add_argument('--num_norm_clusters', type=int, default=10000, required=True,
                        help='number of norm clusters to create via agglomerative clustering')
    args = parser.parse_args()

    called_fun = False

    # Initialize embedder
    embedder = SentenceTransformer('bert-base-nli-mean-tokens')

    # Extract suitable norms
    if args.sc101_path.endswith('.tsv'):
        sc_data_path = construct_json_from_sc(args.sc101_path)
        called_fun = True
    # Obtain norm embeddings
    if args.sc101_path.endswith('.json'):
        sc_data_path = embed_norms(args.sc101_path)
        called_fun = True
    # Cluster and sample norms
    if args.sc101_path.endswith('.embeddings.pkl'):
        cluster_norms(args.sc101_path, args.num_norm_clusters)
        called_fun = True

    if not called_fun:
        print('Unsupported format!')
