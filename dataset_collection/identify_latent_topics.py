import json
import spacy
import string
import gensim
import argparse

from nltk.corpus import stopwords


def _lemmatize(line, remove_stopwords):
    """ Helper function for obtaining various word representations """
    # strip, remove punctuation
    line = line.translate(translator).lower().strip()
    # Remove double space
    line = ' '.join(line.split())
    # Tokenize etc.
    line_nlp = nlp(line)
    spacy_tokens_lower = [elem.text.lower() for elem in line_nlp]
    spacy_lemmas = list()
    for elem in line_nlp:
        if elem.lemma_ == '-PRON-' or elem.lemma_.isdigit():
            spacy_lemmas.append(elem.lower_)
        else:
            spacy_lemmas.append(elem.lemma_.lower().strip())

    if remove_stopwords:
        # Filter out stopwords
        nsw_spacy_lemmas = list()
        for tok_id, tok in enumerate(spacy_tokens_lower):
            if tok not in STOP_WORDS and spacy_lemmas[tok_id] not in STOP_WORDS:
                nsw_spacy_lemmas.append(spacy_lemmas[tok_id])
        spacy_lemmas = nsw_spacy_lemmas

    return spacy_lemmas


def get_topics(train_path, dev_path, test_path, num_topics):
    """ Uses LDA to identify num_topics dominant topics in the dataset. """

    print('Reading-in data ...')
    # Identify names
    names = []
    for dataset_path in [train_path, dev_path, test_path]:
        with open(dataset_path, "r", encoding="utf-8") as dp:
            for line_id, line in enumerate(dp):
                record = json.loads(line)
                name = record['intention'].split()[0].strip().lower()
                if name not in names:
                    names.append(name)

    # Read-in data
    stories = []
    for dataset_path in [train_path, dev_path, test_path]:
        with open(dataset_path, "r", encoding="utf-8") as dp:
            for line_id, line in enumerate(dp):
                record = json.loads(line)
                story = ' '.join([record[k].strip() for k in record.keys() if k != 'qID'])
                lemmas = _lemmatize(story, True)
                lemmas = [lem for lem in lemmas if lem not in names]
                stories.append(lemmas)

    print('Running LDA ...')
    # Create dictionary
    dictionary = gensim.corpora.Dictionary(stories)
    corpus = [dictionary.doc2bow(story) for story in stories]

    # Find topics
    lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    topics = lda_model.print_topics(num_words=5)
    for topic in topics:
        print(topic)


if __name__ == '__main__':
    # Instantiate processing pipeline
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'textcat'])
    # Import stopword list
    STOP_WORDS = [w for w in stopwords.words('english') if w != 'not']

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_split_path', type=str, required=True,
                        help='path to the training data')
    parser.add_argument('--dev_split_path', type=str, required=True,
                        help='path to the development data')
    parser.add_argument('--test_split_path', type=str, required=True,
                        help='path to the test data')
    parser.add_argument('--num_lda_topics',
                        type=int, default=10, help='number of latent topics to discover')
    args = parser.parse_args()

    get_topics(args.train_split_path, args.dev_split_path, args.test_split_path, args.num_lda_topics)
