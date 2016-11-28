import logging as log
import os
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


def make_vocab(nlp, vector_length=-1, ngram=3):
    raw_tags = ('O', 'I', 'B')
    tags = CategoricalTags(raw_tags)

    # processing corpora in realtime and saving it for later use
    corpora = get_corpora()
    encoder = LetterNGramEncoder(nlp, tags, corpora=corpora, vector_length=vector_length, ngram=ngram)
    log.info('Data: Saving vocabulary...')
    encoder.save_vocab()
    log.info('Data: Saved vocabulary. Vector length is {}'.format(encoder.vector_length))


# todo: make one build function that tries to load vocab from default location
def default_encoder(nlp, vocab_path=None):
    raw_tags = ('O', 'I', 'B')
    tags = CategoricalTags(raw_tags)

    if not vocab_path:
        ngram = 3
        vector_length = 30000
        vocab_path = '/home/gkirg/projects/Event-extraction/experiments/ner_tagging/' \
                              'encoder_vocab_{}gram_{}len.bin'.format(ngram, vector_length)
    encoder = LetterNGramEncoder(nlp, tags, path_to_saved_vocab=vocab_path)
    return encoder


def build_encoder(nlp, vector_length, ngram=3):
    raw_tags = ('O', 'I', 'B')
    tags = CategoricalTags(raw_tags)

    # vocab extracted from Open American National Corpus - contemporary American English
    # (~15*10^6 words in dataset overall; around  9.2*10^6 words used)
    # and articles from database
    possible_vocab_path = '/home/gkirg/projects/Event-extraction/experiments/ner_tagging/' \
                          'encoder_vocab_{}gram_{}len.bin'.format(ngram, vector_length)

    if os.path.isfile(possible_vocab_path):
        # loading already processed corpora
        encoder = LetterNGramEncoder(nlp, tags, path_to_saved_vocab=possible_vocab_path,
                                     vector_length=vector_length, ngram=ngram)
        log.info('Data: Loaded vocabulary. Vector length is {}'.format(encoder.vector_length))
        return encoder
    else:
        log.warning('Data: vocab with path {} is not found!'.format(possible_vocab_path))


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
    vector_length = 10000
    nlp = English()
    encoder = build_encoder(nlp, vector_length)
    data_thing = NERPreprocessor()

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


def test_shapes():
    batch_size = 16
    timesteps = 100
    x_len = 30000
    ngram = 3

    nlp = English()
    encoder = build_encoder(nlp, vector_length=x_len, ngram=ngram)
    data_thing = NERPreprocessor()

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

    # test()
    test_shapes()
