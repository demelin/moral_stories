import csv
import argparse
import datetime
import krippendorff

import numpy as np


def show_stats(csv_path):
    """ Displays basic human evaluation statistics. """

    # Relevant fields
    loc_worker_id = 15
    loc_time_spent = 35
    loc_consequence_is_coherent = 32
    loc_consequence_is_likely = 33

    loc_norm = 31
    loc_context = 29
    loc_intention = 30
    loc_action = 27
    loc_consequence = 28

    # Initialize ratings table | {story_id: {category: worker ratings}}
    ratings = dict()
    all_times = list()
    stories = dict()

    story_id = 0
    story_to_id = dict()

    all_worker_ids = list()
    coherent_per_worker = dict()
    incoherent_per_worker = dict()

    # Read in files
    print('Reading-in AMT results ...')
    with open(csv_path, 'r', encoding='utf8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row_id, row in enumerate(csv_reader):
            if row_id == 0:
                continue
            else:
                try:
                    # Parse row
                    norm = row[loc_norm].strip()
                    context = row[loc_context].strip()
                    intention = row[loc_intention].strip()
                    action = row[loc_action].strip()
                    consequence = row[loc_consequence].strip()
                    story_string = ' '.join([norm, context, intention, action, consequence])

                    all_times.append(float(row[loc_time_spent].strip()))
                    worker_id = row[loc_worker_id].strip()
                    consequence_is_coherent = int(row[loc_consequence_is_coherent].strip())
                    if worker_id not in all_worker_ids:
                        all_worker_ids.append(worker_id)

                    if consequence_is_coherent > 3:

                        # Check if certain workers mostly mark consequences as coherent
                        if coherent_per_worker.get(worker_id, None) is None:
                            coherent_per_worker[worker_id] = 0
                        coherent_per_worker[worker_id] += 1

                        consequence_is_likely = int(row[loc_consequence_is_likely].strip())

                    else:

                        # Check if certain workers mostly mark consequences as incoherent
                        if incoherent_per_worker.get(worker_id, None) is None:
                            incoherent_per_worker[worker_id] = 0
                        incoherent_per_worker[worker_id] += 1

                        consequence_is_likely = np.nan
                except ValueError as e:
                    print(e)
                    continue

                # Populate tables
                if story_to_id.get(story_string, None) is None:
                    story_id += 1
                    story_to_id[story_string] = story_id
                    ratings[story_id] = {'consequence_is_coherent': list(),
                                         'consequence_is_coherent_acc': list(),
                                         'consequence_is_likely': list(),
                                         'consequence_is_likely_acc': list()}
                    stories[story_id] = [norm, context, intention, action, consequence]

                if np.isnan(consequence_is_coherent):
                    consequence_is_coherent = 0
                if np.isnan(consequence_is_likely):
                    consequence_is_likely = 0

                ratings[story_id]['consequence_is_coherent'].append(consequence_is_coherent)
                ratings[story_id]['consequence_is_coherent_acc'].append(int(consequence_is_coherent > 3))
                ratings[story_id]['consequence_is_likely'].append(consequence_is_likely)
                ratings[story_id]['consequence_is_likely_acc'].append(int(consequence_is_likely > 3))

    # Identify incoherent stories
    incoherent_stories = list()
    for story_id in ratings.keys():
        if ratings[story_id]['consequence_is_coherent_acc'].count(1) < 2:
            incoherent_stories.append(story_id)

    # Compute means
    category_means = dict()
    for category in ['consequence_is_coherent', 'consequence_is_likely']:
        category_means[category] = list()
        for story_id in ratings.keys():
            if category != 'consequence_is_coherent' and story_id in incoherent_stories:
                category_means[category].append(0.)
            else:
                category_means[category].append(np.mean(ratings[story_id][category]))

    # Compute accuracy
    category_acc = dict()
    for category in ['consequence_is_coherent_acc', 'consequence_is_likely_acc']:
        category_acc[category] = list()
        for story_id in ratings.keys():
            if category != 'consequence_is_coherent_acc' and story_id in incoherent_stories:
                category_acc[category].append(0.)
            else:
                category_acc[category].append(int(ratings[story_id][category].count(1) >= 2))

    # Report (expand as needed)
    print('=' * 20)
    print('Time taken per HIT (mean, std): {}, {}'.format(str(datetime.timedelta(seconds=float(np.mean(all_times)))),
                                                          str(datetime.timedelta(seconds=float(np.std(all_times))))))

    print('-' * 20)
    print('Average story score mean per category:')
    for category in category_means.keys():
        print('{:s}: {:.2f}'.format(category, np.mean(category_means[category])))
    print('\n')
    print('Fraction of valid stories per category:')
    for category in category_acc.keys():
        print('{:s}: {:.2f}'.format(category, np.mean(category_acc[category])))
    print('-' * 20)

    print('-' * 20)
    print('Inter-annotator agreement per category:')
    categories_list = list(ratings[1].keys())

    agreement_values = list()
    for category in categories_list:
        category_arr = [[ratings[s_id][category][0] for s_id in range(1, len(ratings.keys()) + 1)],
                        [ratings[s_id][category][1] for s_id in range(1, len(ratings.keys()) + 1)],
                        [ratings[s_id][category][2] for s_id in range(1, len(ratings.keys()) + 1)]]

        cat_agr = krippendorff.alpha(category_arr)
        if category.endswith('_acc'):
            agreement_values.append(cat_agr)
        print('{:s} : {:.4f}'.format(category, cat_agr))
    print('Mean inter-annotator agreement: {:.4f}'.format(np.mean(agreement_values)))
    print('-' * 20)

    print('-' * 20)
    print('Worker coherence assignments:')
    coh_counts = list()
    for worker_id in all_worker_ids:
        num_coherent = coherent_per_worker.get(worker_id, 0)
        num_incoherent = incoherent_per_worker.get(worker_id, 0)
        coh_counts.append((worker_id, num_coherent, num_incoherent))
    coh_counts = sorted(coh_counts, reverse=True, key=lambda x: x[2])
    for tpl in coh_counts:
        print('Worker {} : {} coherent, {} incoherent'.format(tpl[0], tpl[1], tpl[2]))
    print('-' * 20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratings_file_path', type=str, required=True,
                        help='path to file containing the collected human ratings')
    args = parser.parse_args()

    show_stats(args.ratings_file_path)