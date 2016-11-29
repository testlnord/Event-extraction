import logging as log
from os import path
import spacy
from spacy.en import English
from experiments.ner_tagging.net import NERNet
from experiments.ner_tagging.encoder import LetterNGramEncoder
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


def load_nlp(path_to_model=None, batch_size=16):
    nlp = English()
    ner_net = load_default_ner_net(batch_size=batch_size)
    nlp.pipeline.append(ner_net)
    log.info('load_nlp: adding custom entity (iob) tagger to pipeline')
    return nlp


def load_default_ner_net(batch_size=16):
    tags = CategoricalTags(('O', 'I', 'B'))
    encoder = LetterNGramEncoder.from_vocab_file(tags)
    ner_net = NERNet.from_model_file(encoder=encoder, batch_size=batch_size)
    return ner_net


# dummy func
def classifier_net(something): return 0


# todo:
def deploy_pipeline(nlp):
    data_fetcher = ArticleTextFetch()
    preprocessor = NLPPreprocessor(nlp, min_words_in_sentence=3)
    ner_net = load_default_ner_net()

    # pretty ok, nothing special. anyway, data_fetch is the first defining step.
    for text in data_fetcher.get_old():
        for sent in preprocessor.sents(text):
            # todo: there must be net classifier. encoder must be in the net.
            category = classifier_net(sent)
            if category:
                # todo: there is an entity extraction and construction
                # todo: there is final step: send event to the database
                # that's all
                pass

# todo:
def train_pipeline(nlp):
    data_fetcher = ArticleTextFetch() # let it be articles
    preprocessor = NLPPreprocessor(nlp, min_words_in_sentence=3)
    ner_net = load_default_ner_net()

    raw_tags = (0, 1)
    tags = CategoricalTags(raw_tags)
    tagger1 = HeuristicSpanTagger(tags, nlp)
    tagger2 = TextUserTagger(tags, None)
    tagger = ChainTagger(tags)
    tagger.add_tagger(tagger1)
    tagger.add_tagger(tagger2)

    # todo: constructing classifier net with its' encoder
    encoder = SentenceEncoder(tags)
    classifier = classifier_net

    data_for_later_use = []
    untagged_data = []
    for text in data_fetcher.get_old():
        for sent in preprocessor.sents(text):
            # todo: problem with types
            ner_net(object)

            # todo: what if None?
            tag = tagger.tag(sent)
            if tag:
                # todo: saving tagged data for later use (training)
                data_for_later_use.append((sent, tag))
            else:
                # todo: saving utagged data for later use
                untagged_data.append(sent)

    for data in data_for_later_use:
        # todo: train classifier with that
        pass


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

    path_to_vecs = '/media/Documents/datasets/word_vecs/glove.840B.300d.bin'
    # model_path = 'models/model_full_epochsize{}_epoch{:02d}_valloss{}.h5'.format(8192, 8, 0.23)

    nlp = load_nlp()
    test_nernet(nlp)

