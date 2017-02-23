import logging as log
from spacy.en import English
from experiments.data_common import *
from experiments.marking.tags import CategoricalTags
from experiments.ner_tagging.data_fetcher import NERDataFetcher
from experiments.ner_tagging.encoder import LetterNGramEncoder


def test_single_token(encoder, token, enc=None):
    token_enc = enc if enc is not None else encoder.encode_token(token)
    ngram_mask = token_enc[:-encoder.nb_other_features] # throaw away additional features in encoding

    ngrams_in_token = len(token)
    unique_ngrams = int(sum(ngram_mask))
    is_lost_ngrams = token_enc[-5]

    if unique_ngrams != ngrams_in_token:
        print(is_lost_ngrams, ';', unique_ngrams, '-', ngrams_in_token, ';', token)


def test_print_vocab(encoder, step=1000):
    i = 0
    for i in range(0, encoder.vector_length-step, step):
        print('[{}:{}]'.format(i, i+step))
        print(encoder.vocab[i:i+step])
    print('[{}:{}] (end)'.format(i, len(encoder.vocab)))
    print(encoder.vocab[-step:])
    print('vector_length', encoder.vector_length)


def test_shapes():
    batch_size = 16
    timesteps = 100
    x_len = 30000
    ngram = 3

    nlp = English()
    tags = CategoricalTags(('O', 'I', 'B'))
    encoder = LetterNGramEncoder.from_vocab_file(tags, force_vector_length=x_len)
    data_thing = NERDataFetcher()

    data_fetch = encoder(data_thing._wikigold_conll())
    # path='data_encoded_wikigold_{}gram_{}len.bin'.format(ngram, x_len)
    # data_fetch = deserialize_data_with_logging(path)

    pad = Padding(timesteps)
    batch = BatchMaker(batch_size)
    batched = batch.batch_transposed(pad(data_fetch))

    # for i, b in enumerate(islice(batched, 0, 5)):
    for i, b in enumerate(batched):
        print(i, ': elems:', len(b), 'shape: ({}, {}, {})'.format(*list(map(np.shape, b))))
        x, y, sw = b
        print(type(b))
        print(type(x), type(y), type(sw))
        print(type(x[0]), type(y[0]), type(sw[0]))


if __name__ == '__main__':
    log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)
    test_shapes()
