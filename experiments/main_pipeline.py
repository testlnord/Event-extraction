import logging as log
from os import path
import spacy
from experiments.ner_tagging.net import NERNet
from experiments.ner_tagging.encoder import LetterNGramEncoder
from experiments.marking.data_fetcher import ArticleTextFetch
from experiments.marking.preprocessor import PreprocessTexts
from experiments.marking.tags import CategoricalTags
from experiments.marking.tagger import HeuristicSpanTagger, TextUserTagger
from experiments.marking.encoder import SentenceEncoder


def load_nlp(lang_id='en', path_to_model=None, path_to_vecs=None, batch_size=16):
    args = {}
    if path_to_vecs and path.isfile(path_to_vecs):
        def add_vectors(vocab):
            # vocab.resize_vectors(vocab.load_vectors_from_bin_loc(open(path_to_vecs)))
            vocab.load_vectors_from_bin_loc(path_to_vecs)

        args['add_vectors'] = add_vectors
        log.info('load_nlp: loading custom word vectors from {}'.format(path_to_vecs))
    else:
        log.info('load_nlp: file for word vectors not found; path: {}'.format(path_to_vecs))
        log.info('load_nlp: loading default word vectors')

    if path_to_model and path.isfile(path_to_model):
        def create_pipeline(nlp):
            # todo: there is a cycle because encoder uses nlp there, in the pipeline.
            return [nlp.tagger, nlp.parser, nlp.entity, load_default_ner_net(batch_size)]

        args['create_pipeline'] = create_pipeline
        log.info('load_nlp: adding custom entity (iob) tagger to pipeline')

    nlp = spacy.load(lang_id, **args)
    return nlp


def load_default_ner_net(batch_size=16):
    tags = CategoricalTags(('O', 'I', 'B'))
    encoder = LetterNGramEncoder.from_vocab_file(nlp, tags)
    ner_net = NERNet.from_model_file(encoder=encoder, batch_size=batch_size)
    return ner_net


# todo:
def deploy_pipeline():
    path_to_vecs = '/media/Documents/datasets/word_vecs/glove.840B.300d.bin'
    model_path = 'ner_tagging/models/model_full_epochsize{}_epoch{:02d}_valloss{}.h5'.format(8192, 8, 0.23)
    nlp = load_nlp(path_to_model=model_path)

    data_fetcher = ArticleTextFetch()
    preprocessor = PreprocessTexts(nlp, min_words_in_sentence=3)
    # loading classifier network
    # loading final event_maker


# todo:
def train_pipeline():
    path_to_vecs = '/media/Documents/datasets/word_vecs/glove.840B.300d.bin'
    model_path = 'ner_tagging/models/model_full_epochsize{}_epoch{:02d}_valloss{}.h5'.format(8192, 8, 0.23)
    nlp = load_nlp(path_to_model=model_path)
    pass


def test_nernet(nlp, ner_net):
    with open('samples.txt') as f:
        log.info('Test NERNet: making doc...')
        doc = nlp(f.read())
        ner_net(doc)
        log.info('Test NERNet: made doc')

        for n, sent in enumerate(doc.sents):
            print('\nSENTENCE #{}'.format(n))
            for i, t in enumerate(sent):
                print(str(i).rjust(6), t.ent_type_.ljust(10), t.ent_iob_, t)


if __name__ == "__main__":
    log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)

    path_to_vecs = '/media/Documents/datasets/word_vecs/glove.840B.300d.bin'
    model_path = 'ner_tagging/models/model_full_epochsize{}_epoch{:02d}_valloss{}.h5'.format(8192, 8, 0.23)

    nlp = load_nlp()
    ner_net = load_default_ner_net()
    test_nernet(nlp, ner_net)


