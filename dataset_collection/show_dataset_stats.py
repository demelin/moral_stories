import csv
import argparse
import datetime

import numpy as np
import matplotlib.pyplot as plt


def show_stats(csv_paths, worker_blacklist):
    """ Displays basic statistics of the collected dataset and the workers involved in its construction. """

    # Relevant fields
    loc_worker_id = 15

    loc_prompt1 = 27
    loc_prompt2 = 28
    loc_prompt3 = 29
    loc_prompt_pick = 40

    loc_time_spent = 41
    loc_situation = 37
    loc_intention = 39
    loc_moral_action = 31
    loc_moral_consequence = 35
    loc_immoral_action = 30
    loc_immoral_consequence = 34
    loc_actor_in_situation = 33
    loc_consequence_target = 36

    # Split blacklist
    if worker_blacklist is not None:
        blacklisted = worker_blacklist.strip().split(',')
    else:
        blacklisted = list()

    # Split path string
    split_paths = csv_paths.strip().split(',')

    # Initialize tables
    stat_tables = [dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict()]
    prompts_used = list()

    # Read in files
    print('Reading-in AMT results ...')
    for csv_path in split_paths:

        with open(csv_path, 'r', encoding='utf8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for row_id, row in enumerate(csv_reader):
                if row_id == 0:
                    continue
                else:
                    # Parse row
                    if '1' in row[loc_prompt_pick].strip():
                        prompt = row[loc_prompt1].strip()
                    elif '2' in row[loc_prompt_pick].strip():
                        prompt = row[loc_prompt2].strip()
                    else:
                        prompt = row[loc_prompt3].strip()
                    prompts_used.append(prompt)

                    worker_id = row[loc_worker_id].strip()
                    time_spent = row[loc_time_spent].strip()
                    situation = row[loc_situation].strip()
                    intention = row[loc_intention].strip()
                    moral_action = row[loc_moral_action].strip()
                    moral_consequence = row[loc_moral_consequence].strip()
                    immoral_action = row[loc_immoral_action].strip()
                    immoral_consequence = row[loc_immoral_consequence].strip()
                    actor_in_situation = row[loc_actor_in_situation].strip()
                    consequence_target = row[loc_consequence_target].strip()

                    fields = [time_spent, situation, intention, moral_action, moral_consequence, immoral_action,
                              immoral_consequence, actor_in_situation, consequence_target]

                    # Ignore blacklisted workers
                    if worker_id in blacklisted:
                        continue

                    # Update tables
                    if stat_tables[0].get(worker_id, None) is None:
                        for tbl in stat_tables:
                            tbl[worker_id] = list()
                    for field_id, field in enumerate(fields):
                        stat_tables[field_id][worker_id].append(field)

    # Report (expand as needed)
    # Samples per worker
    sorted_sample_per_worker = \
        sorted([(k, len(v)) for k, v in stat_tables[0].items()], reverse=True, key=lambda x: x[1])
    print('Read-in {:d} samples in total'.format(sum([tpl[1] for tpl in sorted_sample_per_worker])))
    print('Found {:d} unique workers'.format(len(sorted_sample_per_worker)))
    print('Found {:d} unique prompts'.format(len(list(set(prompts_used)))))
    print('HITs completed per worker:')
    for tpl in sorted_sample_per_worker:
        print('{:s} : {:d}'.format(tpl[0], tpl[1]))
    print('Mean samples per worker: {:.2f} (median: {:.2f}, std: {:.2f})'
          .format(np.mean([tpl[1] for tpl in sorted_sample_per_worker]),
                  np.median([tpl[1] for tpl in sorted_sample_per_worker]),
                  np.std([tpl[1] for tpl in sorted_sample_per_worker])))
    print('=' * 10)
    # Print a histogram
    plt.hist([tpl[1] for tpl in sorted_sample_per_worker], 50)
    plt.xlabel('HITs completed')
    plt.ylabel('Number of workers')
    plt.title('HITs submitted per worker')
    plt.grid(True)
    plt.show()

    # Report time spent
    all_times = list()
    for worker_times in stat_tables[0].values():
        all_times += [float(t) for t in worker_times]
    str(datetime.timedelta(seconds=float(np.mean(all_times))))
    print('Time spent per HIT mean: {:s}'.format(str(datetime.timedelta(seconds=float(np.mean(all_times))))))
    print('Time spent per HIT median: {:s}'.format(str(datetime.timedelta(seconds=float(np.median(all_times))))))
    print('=' * 10)

    # Report mean sentence lengths
    sentence_types = \
        ['situation', 'intention', 'moral_action', 'moral_consequence', 'immoral_action', 'immoral_consequence']
    for table_id, table in enumerate(stat_tables[1: -2]):
        all_sentence_lengths = list()
        for worker_sentences in table.values():
            all_sentence_lengths += [len(s.split()) for s in worker_sentences]
        print('Mean {:s} length in tokens: {:.2f}'.format(sentence_types[table_id], np.mean(all_sentence_lengths)))
    print('=' * 10)

    # Report how frequently actor is mentioned in situation
    yes_count = 0
    no_count = 0
    for worker_answers in stat_tables[-2].values():
        for answer in worker_answers:
            if answer == 'yes':
                yes_count += 1
            else:
                no_count += 1
    print('Number of times actor is mentioned in situation: {:d} ({:.2f}%)'
          .format(yes_count, (yes_count / (yes_count + no_count)) * 100))
    print('Number of times actor is NOT mentioned in situation: {:d} ({:.2f}%)'
          .format(no_count, (no_count / (yes_count + no_count)) * 100))
    print('=' * 10)

    # Report how frequently actor is the consequence target
    actor_count = 0
    other_count = 0
    both_count = 0
    for worker_answers in stat_tables[-1].values():
        for answer in worker_answers:
            if answer == 'actor':
                actor_count += 1
            elif answer == 'other':
                other_count += 1
            else:
                both_count += 1
    print('Number of times actor is the consequence target: {:d} ({:.2f}%)'
          .format(actor_count, (actor_count / (actor_count + other_count + both_count)) * 100))
    print('Number of times actor is NOT the consequence target: {:d} ({:.2f}%)'
          .format(other_count, (other_count / (other_count + actor_count + both_count)) * 100))
    print('Number of times actor is one of the consequence targets: {:d} ({:.2f}%)'
          .format(both_count, (both_count / (other_count + actor_count + both_count)) * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_files', type=str, required=True,
                        help='paths to .tsv files containing collected stories; '
                             'should be a single string with individual paths separated by \',\'')
    parser.add_argument('--worker_blacklist', type=str, default=None,
                        help='IDs of AMT-workers whose submissions are to be ignored; '
                             'should be a single string with individual paths separated by \',\'')
    args = parser.parse_args()

    show_stats(args.dataset_files, args.worker_blacklist)
