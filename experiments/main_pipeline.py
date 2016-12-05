import logging as log
from os import path
import pickle
import spacy
from spacy.en import English
from db.db_handler import DatabaseHandler
from event import Event
from experiments.ner_tagging.net import NERNet
from experiments.ner_tagging.encoder import LetterNGramEncoder
from experiments.marking.net import ClassifierNet
from experiments.marking.data_fetcher import ArticleTextFetch
from experiments.marking.preprocessor import NLPPreprocessor
from experiments.marking.tags import CategoricalTags
from experiments.marking.tagger import HeuristicSpanTagger, TextUserTagger, ChainTagger
from experiments.marking.encoder import SentenceEncoder


def load_nlp2(lang_id='en', path_to_model=None, path_to_vecs=None, batch_size=16):
    args = {}
    if path_to_vecs and path.isfile(path_to_vecs):
        def add_vectors(vocab):
            vocab.load_vectors_from_bin_loc(path_to_vecs)

        args['add_vectors'] = add_vectors
        log.info('load_nlp: loading custom word vectors from {}'.format(path_to_vecs))
    else:
        log.info('load_nlp: file for word vectors not found; path: {}'.format(path_to_vecs))
        log.info('load_nlp: loading default word vectors')

    if path_to_model and path.isfile(path_to_model):
        def create_pipeline(nlp):
            ner_net = load_default_ner_net(batch_size=batch_size)
            return [nlp.tagger, nlp.parser, nlp.entity, ner_net]
        args['create_pipeline'] = create_pipeline
        log.info('load_nlp: adding custom entity (iob) tagger to pipeline')

    nlp = spacy.load(lang_id, **args)
    return nlp


def load_nlp(model_path=None, batch_size=16):
    nlp = English()
    ner_net = load_default_ner_net(model_path=model_path, batch_size=batch_size)
    nlp.pipeline.append(ner_net)
    log.info('load_nlp: adding custom entity (iob) tagger to pipeline')
    return nlp


def load_default_ner_net(model_path=None, batch_size=16):
    raw_tags = ('O', 'I', 'B')
    tags = CategoricalTags(raw_tags)
    encoder = LetterNGramEncoder.from_vocab_file(tags)
    ner_net = NERNet.from_model_file(encoder=encoder, batch_size=batch_size, model_path=model_path)
    return ner_net


def load_default_classifier_net(model_path=None, batch_size=16):
    raw_tags = (0, 1)
    tags = CategoricalTags(raw_tags)
    encoder = SentenceEncoder(tags)
    cl_net = ClassifierNet.from_model_file(encoder=encoder, batch_size=batch_size, model_path=model_path)
    return cl_net


def extract_event(sentence, category):
    entity1 = None
    entity2 = None
    action = None
    date = None
    location = None

    # todo: entity1, entity2 and action extraction heuristics
    for token in sentence:
        pass

    event = Event(entity1=entity1, entity2=entity2, action=action, sentence=sentence,
                  date=date, location=location)
    return event


def deploy_pipeline(nlp):
    db_handler = DatabaseHandler()

    data_fetcher = ArticleTextFetch()
    preprocessor = NLPPreprocessor(nlp, min_words_in_sentence=3)
    classifier = load_default_classifier_net()

    # todo: decent article fetcher
    for article, text in data_fetcher.get_old():
        for sent in preprocessor.sents(text):
            category = classifier.predict(sent)
            if category:
                event = extract_event(sentence=sent, category=category)
                # Final step: send event to the database
                info = db_handler.add_event_or_get_id(event, article)
                log.info('Main_deploy: db_handler: url={}; {}'.format(article.url, info))


# todo: generate dataset
# todo: train classifier net
def train_pipeline(nlp):
    data_fetcher = ArticleTextFetch() # let it be articles
    preprocessor = NLPPreprocessor(nlp, min_words_in_sentence=3)

    raw_tags = (0, 1)
    tags = CategoricalTags(raw_tags)
    # todo: precise heuristic tagger
    tagger1 = HeuristicSpanTagger(tags, nlp)
    # todo: check text user tagger
    tagger2 = TextUserTagger(tags, None)
    tagger = ChainTagger(tags)
    tagger.add_tagger(tagger1)
    tagger.add_tagger(tagger2)

    tagged_data_filename = ClassifierNet.relpath('data', 'data_tagged.pck')
    untagged_data_filename = ClassifierNet.relpath('data', 'data_untagged.pck')
    tagged_data = open(tagged_data_filename, 'wb')
    untagged_data = open(untagged_data_filename, 'wb')

    for article, text in data_fetcher.get_old():
        for sent in preprocessor.sents(text): # there is a ner_net in preprocessor
            tag = tagger.tag(sent)
            if tag is not None:
                # Saving tagged text for later use (training)
                pickle.dump((sent.text, tag), tagged_data)
            else:
                pickle.dump(sent.text, tagged_data)

    tagged_data.close()
    untagged_data.close()


def test_nernet(nlp):
    with open('samples.txt') as f:
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

