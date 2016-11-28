import logging as log
from os import path
import spacy
from experiments.ner_tagging.net import NERNet


def load_nlp(lang_id='en', path_to_model=None, path_to_vecs=None):
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
        # todo: think about these paramteres: how to specify them, where to place them
        vector_length = 30000
        timesteps = 150
        # predict in fixed batches? make sure last possibly incomplete batch is used.
        batch_size = 16

        ner_net = NERNet(x_len=vector_length, batch_size=batch_size, nbclasses=3, timesteps=timesteps,
                         path_to_model=path_to_model)
        def create_pipeline(nlp):
            return [nlp.tagger, nlp.parser, nlp.entity, ner_net]
            # return [nlp.tagger, nlp.parser, ner_net]

        args['create_pipeline'] = create_pipeline
        log.info('load_nlp: adding custom entity (iob) tagger to pipeline')

    nlp = spacy.load(lang_id, **args)
    return nlp


def main():
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
    model_path = 'ner_tagging/models/model_full_epochsize{}_epoch{:02d}_valloss{}.h5'.format(8192, 8, 0.23)
    nlp = load_nlp(path_to_model=model_path)

    test_nernet(nlp)


