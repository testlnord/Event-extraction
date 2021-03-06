import logging as log
import pickle

from spacy.en import English

from db.db_handler import DatabaseHandler
from experiments.event_extractor import EventExtractor
from experiments.marking.article_fetcher import ArticleTextFetcher
from experiments.marking.net import ClassifierNet
from experiments.marking.nlp_preprocessor import NLPPreprocessor
from experiments.marking.sentence_encoder import SentenceEncoder
from experiments.marking.taggers import HeuristicSpanTagger, TextUserTagger, ChainTagger
from experiments.tags import CategoricalTags


def load_default_classifier_net(model_path=None, batch_size=16):
    raw_tags = (0, 1)
    tags = CategoricalTags(raw_tags)
    encoder = SentenceEncoder(tags)
    cl_net = ClassifierNet.from_model_file(encoder=encoder, batch_size=batch_size, model_path=model_path)
    return cl_net


def deploy_pipeline(nlp):
    db_handler = DatabaseHandler()

    data_fetcher = ArticleTextFetcher()
    preprocessor = NLPPreprocessor(nlp, min_words_in_sentence=3)
    classifier = load_default_classifier_net()
    eextractor = EventExtractor()

    # todo: decent article fetcher
    for article, text in data_fetcher.get_old():
        for sent in preprocessor.sents(text):
            category = classifier.predict(sent)
            if category:
                event = eextractor.extract_event(sentence=sent, category=category)
                # Final step: send event to the database
                info = db_handler.add_event_or_get_id(event, article)
                log.info('Main_deploy: db_handler: url={}; {}'.format(article.url, info))


# todo: processing previously untagged data
def process_untagged(nlp):
    pass

# todo: generate dataset
# todo: train classifier net
def train_pipeline(nlp):
    data_fetcher = ArticleTextFetcher() # let it be articles
    preprocessor = NLPPreprocessor(nlp, min_words_in_sentence=3)

    raw_tags = (0, 1)
    tags = CategoricalTags(raw_tags)
    # todo: precise heuristic tagger
    tagger1 = HeuristicSpanTagger(tags, nlp)
    tagger2 = TextUserTagger(tags, None)
    tagger = ChainTagger(tags)
    tagger.add_tagger(tagger1)
    tagger.add_tagger(tagger2)

    tagged_data_filename = ClassifierNet.relpath('data', 'data_tagged.pck')
    untagged_data_filename = ClassifierNet.relpath('data', 'data_untagged.pck')
    tagged_data = open(tagged_data_filename, 'wb')
    untagged_data = open(untagged_data_filename, 'wb')

    i = 0 # Monitoring number of processed data units
    try:

        for article, text in data_fetcher.get_old():
            for sent in preprocessor.sents(text): # there is a ner_net in preprocessor
                tag = tagger.tag(sent)
                i += 1
                if tag is not None:
                    # Saving tagged text for later use (training)
                    pickle.dump((sent.text, tag), tagged_data)
                else:
                    pickle.dump(sent.text, tagged_data)

    # This allows to continue tagging next time exactly from the place we stopped
    except KeyboardInterrupt:
        print('train_pipeline: interrupt: stopping at data unit №{} (counting from zero)'.format(i))
    except Exception as e:
        print('train_pipeline: exception occured at data unit №{} (counting from zero)'.format(i))
        raise
    else:
        tagged_data.close()
        untagged_data.close()


def test_nernet(nlp, data_file='data/samples.txt'):
    with open(data_file) as f:
        log.info('Test NERNet: making doc...')
        doc = nlp(f.read())
        log.info('Test NERNet: made doc')

        for n, sent in enumerate(doc.sents):
            print('\nSENTENCE #{}'.format(n))
            for i, t in enumerate(sent):
                print(str(i).rjust(6), t.ent_type_.ljust(10), t.ent_iob_, t)


if __name__ == "__main__":
    log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)

    nlp = load_nlp()
    test_nernet(nlp)

