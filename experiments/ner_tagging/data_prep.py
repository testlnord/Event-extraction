import logging as log
import pickle
import os
from itertools import islice, cycle
from spacy.en import English
from experiments.data_common import *
from experiments.marking.tags import CategoricalTags
from experiments.marking.data_fetcher import FilesFetcher, ArticleTextFetch
from experiments.ner_tagging.preprocessor import NERPreprocessor
from experiments.ner_tagging.encoder import LetterNGramEncoder


def get_corpora(root_dir_of_corpora_files='/media/Documents/datasets/OANC-GrAF/data/written_2'):
    total_texts = 0
    total_symbols = 0

    min_len = 5
    articles_fetcher = ArticleTextFetch(return_header=True, return_summary=True, return_article_text=True)
    for text in articles_fetcher.get_old():
        if len(text) >= min_len:
            total_symbols += len(text)
            total_texts += 1
            yield text

    texts = FilesFetcher.get_texts_from_filetree(root_dir_of_corpora_files)
    for text in texts:
        if len(text) >= min_len:
            total_symbols += len(text)
            total_texts += 1
            yield text

    log.info('get_corpora: total: texts='
             '{}, symbols={}, approx. words={}'.format(total_texts, total_symbols, total_symbols/5.1))


def make_vocab(vector_length=-1, ngram=3):
    nlp = English()
    log.info('Data: Loaded spacy')
    raw_tags = ('O', 'I', 'B')
    tags = CategoricalTags(raw_tags)

    # processing corpora in realtime and saving it for later use
    corpora = get_corpora()
    encoder = LetterNGramEncoder(nlp, tags, corpora=corpora, vector_length=vector_length, ngram=ngram)
    log.info('Data: Saving vocabulary...')
    encoder.save_vocab()
    log.info('Data: Saved vocabulary. Vector length is {}'.format(encoder.vector_length))


def data_gen(vector_length, ngram=3):
    nlp = English()
    log.info('Data: Loaded spacy')
    raw_tags = ('O', 'I', 'B')
    tags = CategoricalTags(raw_tags)

    # vocab extracted from Open American National Corpus - contemporary American English
    # (~15*10^6 words in dataset overall; around  9.2*10^6 words used)
    # and articles from database
    possible_vocab_path = '/home/gkirg/projects/Event-extraction/experiments/ner_tagging/' \
                          'encoder_vocab_{}gram_{}len.bin'.format(ngram, vector_length)

    if os.path.isfile(possible_vocab_path):
        data_thing = NERPreprocessor()
        # loading already processed corpora
        encoder = LetterNGramEncoder(nlp, tags, path_to_saved_vocab=possible_vocab_path,
                                     vector_length=vector_length, ngram=ngram)
        log.info('Data: Loaded vocabulary. Vector length is {}'.format(encoder.vector_length))
        return data_thing, encoder
    else:
        log.warning('Data: vocab with path {} is not found!'.format(possible_vocab_path))


def serialize_data(iterable_data, output_path='data_encoded.bin'):
    with open(output_path, 'wb') as f:
        for i, data_chunk in enumerate(iterable_data):
            log.info('serialise_data: data_chunk #{}'.format(i))
            pickle.dump(data_chunk, f)


def deserialize_data(file_path='data_encoded.bin'):
    with open(file_path, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def deserialize_data_with_logging(file_path='data_encoded.bin'):
    i = 0
    with open(file_path, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
                log.info('deserialise_data: data_chunk #{}'.format(i))
                i += 1
            except EOFError:
                log.info('deserialise_data: total: {}'.format(i))
                break


def test_serialisation(nb_items=50):
    # iterable = ((i, i**2, i**3) for i in range(nb_items))
    data_thing, encoder = data_gen()
    iterable = islice(encoder(data_thing.objects()), 0, nb_items)
    serialize_data(iterable)
    for el in deserialize_data_with_logging():
        print(el)


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


def test():
    data_thing, encoder = data_gen(10000)

    # test 1
    test_tokens = ['a', 'ab', 'abe', 'xxxxx', 'cat', 'aaaaaaa', 'banana', 'webinar']
    for token in test_tokens:
        test_single_token(encoder, token)

    # test 2
    data_fetch = data_thing.objects()
    start = 0
    end = 131072
    print('STARTED')
    for i, (sent, tags) in enumerate(islice(data_fetch, start, end)):
        print('test: encoding #{}'.format(i), sent)
        tags_enc = np.array(encoder.encode_tags(tags))
        sent_enc = np.array(encoder.encode_text(sent))

        # for token, token_enc in zip(sent, sent_enc):
        #     test_single_token(encoder, token, token_enc)
    print('FINISHED')


def test_mem():
    """Very bad method... eats memory instantly"""
    data_thing, encoder = data_gen(30000)
    batch = BatchMaker(32)
    padder = Padding(150)
    data_fetch = data_thing.objects()

    print('STARTED')
    for i, b in enumerate(cycle(batch(padder(encoder(data_fetch))))):
        print('test: batch #{}'.format(i))
    print('FINISHED')


def test_mem2():
    data_thing, encoder = data_gen(30000)
    batch = BatchMaker(32)
    padder = Padding(150)

    def data_split():
        return batch(padder(
            encoder(data_thing.objects())))

    print('STARTED')
    for i, b in enumerate(cycle_uncached(data_split)):
        print('test: batch #{}'.format(i))
    print('FINISHED')


def test_shapes():
    batch_size = 16
    timesteps = 100
    x_len = 30000
    ngram = 3

    data_thing, encoder = data_gen(x_len)
    data_fetch = encoder(data_thing._wikigold_conll())
    # path='data_encoded_wikigold_{}gram_{}len.bin'.format(ngram, x_len)
    # data_fetch = deserialize_data_with_logging(path)

    pad = Padding(timesteps)
    batch = BatchMaker(batch_size)
    batched = batch(pad(data_fetch))

    # for i, b in enumerate(islice(batched, 0, 5)):
    for i, b in enumerate(batched):
        print(i, ': elems:', len(b), 'shape: ({}, {}, {})'.format(*list(map(np.shape, b))))
        x, y, sw = b
        print(type(b))
        print(type(x), type(y), type(sw))
        print(type(x[0]), type(y[0]), type(sw[0]))


if __name__ == '__main__':
    log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)
    # test()
    test_shapes()
    # test_serialisation()
    # test_mem()

    # data_thing, encoder = data_gen(10000)
    # for i, item in enumerate(encoder(data_thing._wikigold_conll()), 1):
    #     print(i)

    # data_wikigold = encoder(data_thing._wikigold_conll())
    # serialize_data(data_wikigold,
    #                output_path='data_encoded_wikigold_{}gram_{}len.bin'.format(encoder.ngram, encoder.vector_length))
    # data_wikiner = encoder(data_thing._wikiner_wp3())
    # serialize_data(data_wikiner,
    #                output_path='data_encoded_wikiner_{}gram_{}len.bin'.format(encoder.ngram, encoder.vector_length))

