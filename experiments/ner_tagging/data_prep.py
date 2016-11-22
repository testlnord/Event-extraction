import logging as log
import pickle
import os
from itertools import islice
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


def data_gen(vector_length=-1, ngram=3):
    default_vector_length = 10000
    nlp = English()
    log.info('Data: Loaded spacy')
    raw_tags = ('O', 'I', 'B')
    tags = CategoricalTags(raw_tags)

    # vocab extracted from Open American National Corpus - contemporary American English
    # (~15*10^6 words in dataset overall; around  9.2*10^6 words used)
    # and articles from database
    # todo: temp
    # len_to_load = vector_length if vector_length > 0 else default_vector_length
    len_to_load = default_vector_length
    possible_vocab_path = 'encoder_vocab_{}gram_{}len.bin'.format(ngram, len_to_load)

    data_thing = NERPreprocessor()
    encoder = None
    if os.path.exists(possible_vocab_path):
        # loading already processed corpora
        encoder = LetterNGramEncoder(nlp, tags, path_to_saved_vocab=possible_vocab_path,
                                     vector_length=vector_length, ngram=ngram)
        log.info('Data: Loaded vocabulary. Vector length is {}'.format(encoder.vector_length))
    else:
        # processing corpora in realtime and saving it for later use
        corpora = get_corpora()
        encoder = LetterNGramEncoder(nlp, tags, corpora=corpora, vector_length=vector_length, ngram=ngram)
        log.info('Data: Saving vocabulary...')
        encoder.save_vocab()
        log.info('Data: Saved vocabulary. Vector length is {}'.format(encoder.vector_length))

    return data_thing, encoder


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
    # test_print_vocab(encoder)

    # test 1
    test_tokens = ['a', 'ab', 'abe', 'xxxxx', 'cat', 'aaaaaaa', 'banana', 'webinar']
    for token in test_tokens:
        test_single_token(encoder, token)

    # test 2
    # data_fetch = data_thing._wikigold_conll()
    data_fetch = data_thing.objects()
    # start = 65536
    start = 0
    end = 131072
    print('STARTED')
    for i, (sent, tags) in enumerate(islice(data_fetch, start, end)):
        print('test: encoding #{}'.format(i), sent)
        tags_enc = np.array(encoder.encode_tags(tags))
        sent_enc = np.array(encoder.encode_text(sent))
        # print(sent)
        # for token, token_enc in zip(sent, sent_enc):
        #     test_single_token(encoder, token, token_enc)
    print('FINISHED')


def test_gen():
    data_thing, encoder = data_gen(10000)
    data_fetch = encoder(data_thing.objects())
    data_fetch = data_thing.objects()

    slice_size = 10
    for i in range(1, 5):
        for j, (sent, tag) in enumerate(islice(data_fetch, 0, slice_size)):
            print(i, j, sent)


def test_shapes():
    batch_size = 16
    timesteps = 100
    x_len = 10000
    ngram = 3

    data_thing, encoder = data_gen(10000)
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
    test_shapes()
    # test()
    # test_serialisation()
    # test_gen()

    # data_thing, encoder = data_gen(10000)
    # for i, item in enumerate(encoder(data_thing._wikigold_conll()), 1):
    #     print(i)

    # data_wikigold = encoder(data_thing._wikigold_conll())
    # serialize_data(data_wikigold,
    #                output_path='data_encoded_wikigold_{}gram_{}len.bin'.format(encoder.ngram, encoder.vector_length))
    # data_wikiner = encoder(data_thing._wikiner_wp3())
    # serialize_data(data_wikiner,
    #                output_path='data_encoded_wikiner_{}gram_{}len.bin'.format(encoder.ngram, encoder.vector_length))

