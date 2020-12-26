import csv
import argparse


def filter_stories(stories_path, validation_path):
    """ Filters out moral stories that are poorly rated by human judges from the overall set of collected stories. """

    # Relevant fields
    loc_story_id = 35
    loc_actions_fulfill_intention = 37
    loc_actions_are_aligned = 36
    loc_consequences_are_reasonable = 38

    # Initialize ratings table | {story_id: {category: worker ratings}}
    ratings = dict()
    stories_to_keep = list()

    # Read in files
    print('Reading-in AMT results ...')
    with open(validation_path, 'r', encoding='utf8') as val_file:
        val_rows = csv.reader(val_file, delimiter=',')
        for row_id, row in enumerate(val_rows):
            if row_id == 0:
                continue
            else:
                try:
                    # Parse row
                    story_id = int(row[loc_story_id].strip())
                    actions_fulfill_intention = int(row[loc_actions_fulfill_intention].strip())
                    actions_are_aligned = int(row[loc_actions_are_aligned].strip())
                    consequences_are_reasonable = int(row[loc_consequences_are_reasonable].strip())
                except ValueError as e:
                    print(e)
                    print(row_id, row)
                    continue

                # Populate tables
                if ratings.get(story_id, None) is None:
                    ratings[story_id] = {'actions_fulfill_intention': actions_fulfill_intention,
                                         'actions_are_aligned': actions_are_aligned,
                                         'consequences_are_reasonable': consequences_are_reasonable}

    # Check if story is to be rejected
    for story_id in ratings.keys():
        drop_story = False
        # Drop a story if at least one category received a low score
        for category in ratings[story_id]:
            if ratings[story_id][category] < 4:
                drop_story = True
                break
        if not drop_story:
            stories_to_keep.append(story_id)

    # Write highly-rated stories to a separate file
    out_path = stories_path[:-4] + '_validated.tsv'
    with open(stories_path, 'r', encoding='utf8') as stories_file:
        with open(out_path, 'w', encoding='utf8') as out_file:
            for row_id, row in enumerate(stories_file):
                if row_id == 0:
                    out_file.write(row)
                else:
                    if row_id in stories_to_keep:
                        out_file.write(row)

    print('Wrote highly-rated stories to {:s}'.format(out_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--moral_stories_path', type=str, required=True,
                        help='path to file containing all collected moral stories')
    parser.add_argument('--validation_scores_path', type=str, required=True,
                        help='path to file containing human acceptability ratings for each story')
    args = parser.parse_args()

    filter_stories(args.moral_stories_path, args.validation_scores_path)
