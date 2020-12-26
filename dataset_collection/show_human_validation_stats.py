import csv
import argparse
import datetime

import numpy as np
import matplotlib.pyplot as plt


def show_stats(csv_paths):
    """ Displays basic human validation statistics. """

    # Relevant fields
    loc_worker_id = 15
    loc_time_spent = 43
    loc_story_id = 35
    loc_actions_fulfill_intention = 37
    loc_actions_are_aligned = 36
    loc_consequences_are_reasonable = 38
    sample_story_id = None

    loc_norm = 28
    loc_situation = 29
    loc_intention = 30
    loc_moral_action = 31
    loc_moral_consequence = 32
    loc_immoral_action = 33
    loc_immoral_consequence = 34

    # Initialize ratings table | {story_id: {category: worker ratings}}
    ratings = dict()
    all_times = list()
    stories = dict()
    workers = dict()
    worker_times = dict()

    # Split path string
    split_paths = csv_paths.strip().split(',')

    # Read in files
    print('Reading-in AMT results ...')
    for csv_path in split_paths:
        with open(csv_path, 'r', encoding='utf8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row_id, row in enumerate(csv_reader):
                if row_id == 0:
                    continue
                else:
                    try:
                        # Parse row
                        all_times.append(float(row[loc_time_spent].strip()))
                        story_id = row[loc_story_id].strip()
                        worker_id = row[loc_worker_id].strip()
                        actions_fulfill_intention = int(row[loc_actions_fulfill_intention].strip())
                        actions_are_aligned = int(row[loc_actions_are_aligned].strip())
                        consequences_are_reasonable = int(row[loc_consequences_are_reasonable].strip())
                    except ValueError as e:
                        print(e)
                        print(row_id, row)
                        continue

                    # Populate tables
                    if ratings.get(story_id, None) is None:
                        sample_story_id = story_id
                        ratings[story_id] = {'actions_fulfill_intention': [(worker_id, actions_fulfill_intention)],
                                             'actions_are_aligned': [(worker_id, actions_are_aligned)],
                                             'consequences_are_reasonable': [(worker_id, consequences_are_reasonable)]}

                        stories[story_id] = [row[loc_norm].strip(),
                                             row[loc_situation].strip(),
                                             row[loc_intention].strip(),
                                             row[loc_moral_action].strip(),
                                             row[loc_moral_consequence].strip(),
                                             row[loc_immoral_action].strip(),
                                             row[loc_immoral_consequence].strip()]

                    # Update worker statistics
                    if workers.get(worker_id, None) is None:
                        workers[worker_id] = 0
                    workers[worker_id] += 1

                    if worker_times.get(worker_id, None) is None:
                        worker_times[worker_id] = list()
                    worker_times[worker_id].append(float(row[loc_time_spent].strip()))

    # Check if story is to be rejected
    good_count_total = 0
    bad_count_total = 0
    stories_kept = 0
    stories_dropped = 0
    for story_id in ratings.keys():
        dropped = False
        for category in ratings[story_id]:
            # Add categories to be ignored here
            if category in []:
                continue

            # Drop a story if at least one category received a low score
            bad_count = 0
            for r in ratings[story_id][category]:
                if r[1] < 4:
                    bad_count += 1
                    bad_count_total += 1
            if bad_count >= 1:
                dropped = True
                stories_dropped += 1
                break
            else:
                good_count_total += 1

        if not dropped:
            stories_kept += 1
        else:
            print('-' * 20)
            print('DROPPED:')
            print(ratings[story_id])
            print('\n')
            print('NORM: {:s}'.format(stories[story_id][0]))
            print('SITUATION: {:s}'.format(stories[story_id][1]))
            print('INTENTION: {:s}'.format(stories[story_id][2]))
            print('MORAL ACTION: {:s}'.format(stories[story_id][3]))
            print('MORAL CONSEQUENCE: {:s}'.format(stories[story_id][4]))
            print('IMMORAL ACTION: {:s}'.format(stories[story_id][5]))
            print('IMMORAL CONSEQUENCE: {:s}'.format(stories[story_id][6]))

    # Remove worker IDs from ratings
    for story_id in ratings.keys():
        for cat in ratings[story_id].keys():
            scores_only = list()
            for _, r in ratings[story_id][cat]:
                scores_only.append(r)
            ratings[story_id][cat] = scores_only

    # Collect stats
    category_means = {v: list() for v in ratings[list(ratings.keys())[0]]}
    category_stds = {v: list() for v in ratings[list(ratings.keys())[0]]}
    category_mins = {v: list() for v in ratings[list(ratings.keys())[0]]}

    for story_id in ratings.keys():
        for category in ratings[story_id]:
            category_means[category].append(np.mean(ratings[story_id][category]))
            category_stds[category].append(np.std(ratings[story_id][category]))
            category_mins[category].append(min(ratings[story_id][category]))

    # Report (expand as needed)
    print('=' * 20)
    print('Time taken per HIT (mean, std): {}, {}'.format(str(datetime.timedelta(seconds=float(np.mean(all_times)))),
                                                          str(datetime.timedelta(seconds=float(np.std(all_times))))))

    print('-' * 20)
    print('Found ratings for {:d} stories'.format(len(ratings.keys())))
    print('Stories kept: {:d}'.format(stories_kept))
    print('Stories rejected: {:d}'.format(stories_dropped))
    print('\n')

    print('-' * 20)
    print('Average story score mean per category:')
    for category in category_means.keys():
        print('{:s}: {:.2f}'.format(category, np.mean(category_means[category])))
    print('\n')
    print('Average story score std per category:')
    for category in category_stds.keys():
        print('{:s}: {:.2f}'.format(category, np.mean(category_stds[category])))
    print('\n')
    print('Average story score mins per category:')
    for category in category_mins.keys():
        print('{:s} : {:.2f}'.format(category, np.mean(category_mins[category])))
    print('\n')

    print('-' * 20)
    print('Good votes total: {:d}'.format(good_count_total))
    print('Bad votes total: {:d}'.format(bad_count_total))

    print('-' * 20)
    print('# stories rated per worker:')
    workers_sorted = sorted([(w, c) for w, c in workers.items()], reverse=True, key=lambda x: x[1])
    for w, c in workers_sorted:
        print('{:s} : {:d}, {:s}'.format(w, c, str(datetime.timedelta(seconds=float(np.mean(worker_times[w]))))))

    # Plot histogram
    categories_list = list(ratings[sample_story_id].keys())
    category_ratings = [list(), list(), list(), list(), list(), list(), list(), list(), list()]
    for c_id, c in enumerate(categories_list):
        for story_id in ratings.keys():
            category_ratings[c_id] += ratings[story_id][c]

    ones = [lst.count(1) for lst in category_ratings]
    twos = [lst.count(2) for lst in category_ratings]
    threes = [lst.count(3) for lst in category_ratings]
    fours = [lst.count(4) for lst in category_ratings]
    fives = [lst.count(5) for lst in category_ratings]

    ind = np.arange(9)
    width = 0.15
    plt.bar(ind, ones, width, label='1')
    plt.bar(ind + width, twos, width, label='2')
    plt.bar(ind + width * 2, threes, width, label='3')
    plt.bar(ind + width * 3, fours, width, label='4')
    plt.bar(ind + width * 4, fives, width, label='5')

    plt.ylabel('Number of ratings')
    plt.title('Score distribution per category')

    x_label_dict = dict()
    x_label_dict['actions_fulfill_intention'] = 'Actions fulfill intention'
    x_label_dict['actions_are_aligned'] = 'Actions are aligned with moral norm'
    x_label_dict['consequences_are_reasonable'] = 'Consequences are plausible'

    categories_list_plot = [x_label_dict[c] for c in categories_list]

    plt.xticks(ind + width * 2, categories_list_plot, rotation=75)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation_scores_path', type=str, required=True,
                        help='path to file containing human acceptability ratings for each story; '
                             'should be a single string with individual paths separated by \',\'')
    args = parser.parse_args()
    
    show_stats(args.validation_scores_path)
