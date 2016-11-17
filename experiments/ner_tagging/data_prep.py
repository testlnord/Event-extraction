import logging as log
import os
from itertools import islice
from spacy.en import English
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
    nlp = English()
    log.info('Loaded spacy')
    raw_tags = ('O', 'I', 'B')
    tags = CategoricalTags(raw_tags)

    # vocab extracted from Open American National Corpus - contemporary American English
    # (~15*10^6 words in dataset overall; around  9.2*10^6 words used)
    # and articles from database
    len_to_load = vector_length if vector_length > 0 else 30000
    possible_vocab_path = 'encoder_vocab_{}gram_{}len.bin'.format(ngram, len_to_load)

    data_thing = NERPreprocessor()
    encoder = None
    if os.path.exists(possible_vocab_path):
        # loading already processed corpora
        encoder = LetterNGramEncoder(nlp, tags, path_to_saved_vocab=possible_vocab_path,
                                     vector_length=vector_length, ngram=ngram)
        log.info('Loaded vocabulary. Vector length is {}'.format(encoder.vector_length))
    else:
        # processing corpora in realtime and saving it for later use
        corpora = get_corpora()
        encoder = LetterNGramEncoder(nlp, tags, corpora=corpora, vector_length=vector_length, ngram=ngram)
        log.info('Saving vocabulary...')
        encoder.save_vocab()
        log.info('Saved vocabulary. Vector length is {}'.format(encoder.vector_length))

    return data_thing, encoder


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
    data_thing, encoder = data_gen()
    # test_print_vocab(encoder)

    # test 1
    test_tokens = ['a', 'ab', 'abe', 'xxxxx', 'cat', 'aaaaaaa', 'banana', 'webinar']
    for token in test_tokens:
        test_single_token(encoder, token)

    # test 2
    for i, (sent, tags) in enumerate(data_thing.objects()):
        sent_enc, tags_enc = encoder.encode(sent, tags)
        print(sent)
        for token, token_enc in zip(sent, sent_enc):
            test_single_token(encoder, token, token_enc)

# todo: test encoder (+++tag_encoding, ???sentence encoding)
# todo: ~~~refine preprocessor---datafetcher
# todo: train code (batching, dataset splitting)
if __name__ == '__main__':
    log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)
    test()

