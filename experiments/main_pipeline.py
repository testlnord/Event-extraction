import logging as log
from os import path
import spacy
from experiments.ner_tagging.net import NERNet


# todo: test
def load_nlp(lang_id='en', path_to_model=None, path_to_vecs=None):
    args = {}
    if path.isfile(path_to_vecs):
        def add_vectors(vocab):
            # vocab.resize_vectors(vocab.load_vectors_from_bin_loc(open(path_to_vecs)))
            vocab.load_vectors_from_bin_loc(path_to_vecs)

        args['add_vectors'] = add_vectors
        log.info('load_nlp: loading custom word vectors from {}'.format(path_to_vecs))
    else:
        log.info('load_nlp: file for word vectors not found! path: {}'.format(path_to_vecs))
        log.info('load_nlp: loading default word vectors')

    if path.isfile(path_to_model):
        # todo: think about these paramteres: how to specify them, where to place them
        vector_length = 30000
        timesteps = 100
        # predict in fixed batches? make sure last possibly unfull batch is used.
        batch_size = 32

        ner_net = NERNet(vector_length, timesteps=timesteps, batch_size=batch_size,
                         path_to_model=path_to_model)
        def create_pipeline(nlp):
            return [nlp.tagger, nlp.parser, nlp.entity, ner_net]

        args['create_pipeline'] = ner_net
        log.info('load_nlp: adding custom entity (iob) tagger to pipeline')

    nlp = spacy.load(lang_id, **args)
    return nlp


def main():
    pass


if __name__ == "__main__":
    log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)
    path_to_vecs = '/media/Documents/datasets/word_vecs/glove.840B.300d.bin'
    path_to_model = 'ner_tagging/model_full.h5'
    nlp = load_nlp(path_to_model=path_to_model)
