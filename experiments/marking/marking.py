import logging as log
import numpy as np
from spacy.en import English, Vocab
from spacy.tokens import Token, Span

from data_mining.downloaders import article_downloaders
from data_mining.downloaders import blogs_parser, developerandeconomics, infoworld, itnews, slashdot

from experiments.marking.data_fetcher import *
from experiments.marking.preprocessor import *
from experiments.marking.tagger import *
from experiments.marking.tags import *
from experiments.marking.encoder import *


class EventToSpanWrapper:
    def __init__(self, nlp):
        self.nlp = nlp

    def __call__(self, event_tag_pairs):
        for event, tag in event_tag_pairs:
            yield self.nlp(event.sentence)[:], tag

def test_tagger_events():
    nlp = English()
    log.info('Loaded spacy')
    raw_tags = (0, 1)
    tags = CategoricalTags(raw_tags)

    # testing OpenIE
    data_fetcher = FileLineFetcher('../samples-json-relations.txt', nlp)
    preprocessor = PreprocessJsonOpenIEExtractions(nlp, nb_most_confident=1)
    wrapper = EventToSpanWrapper(nlp)

    # encoder = PaddingEncoder(nlp, tags, 30)
    encoder = Encoder(nlp, tags)
    tagger1 = HeuristicEventTagger(tags, nlp)
    tagger2 = DummyTagger(tags, '1')
    tagger = ChainTagger(tags)
    tagger.add_tagger(tagger1)
    tagger.add_tagger(tagger2)

    try:
        data_generator = tagger(preprocessor(data_fetcher.get_raw_lines()))
        for event, tag in data_generator:
            print(tag, event.sentence)

    except KeyboardInterrupt:
        log.info('Keyboard Interrupt happened. Exiting.')
        exit(0)


def get_nlp(path_to_vecs='/media/Documents/datasets/word_vecs/glove.840B.300d.bin'):
    def add_vectors(vocab):
        #vocab.resize_vectors(vocab.load_vectors_from_bin_loc(open(path_to_vecs)))
        vocab.load_vectors_from_bin_loc(path_to_vecs)
    nlp = English(add_vectors=add_vectors)
    log.info('Loaded {} word vectors from {}'.format(len(nlp.vocab), path_to_vecs))
    return nlp


def test_tagger_texts():
    nlp = get_nlp()
    # nlp = English()
    log.info('Loaded spacy')
    raw_tags = (0, 1)
    tags = CategoricalTags(raw_tags)

    drop_file = open('../untagged_last.txt', 'a')
    data0 = ArticleTextFetch().get_old()
    data1 = FileLineFetcher('../samples.txt', nlp).get_raw_lines()
    data2 = FilesFetcher.get_texts_from_files(['../texts.txt']) # Sherlock Holmes and something else
    data3 = FilesFetcher.get_texts_from_filetree('/media/Documents/datasets/OANC-GrAF') # lots of data...

    preprocessor = PreprocessTexts(nlp, min_words_in_sentence=3)
    # encoder = PaddingEncoder(nlp, tags, 50)

    suspicious_ne_types=('ORG', 'PRODUCT', 'FAC', 'NORP', 'EVENT')
    tagger1 = HeuristicSpanTagger(tags, nlp, suspicious_ne_types=suspicious_ne_types)
    tagger2 = DummyTagger(tags, None)
    tagger = ChainTagger(tags)
    tagger.add_tagger(tagger1)
    tagger.add_tagger(tagger2)

    i = 0
    zeros = 0
    try:
        data_generator = tagger(preprocessor(data0))
        for i, (sent, tag) in enumerate(data_generator, 1):
            if tag == tags.default_tag:
                print(i, ':', tag, sent)
                zeros += 1
            elif tag is None:
                # write to file for further analysis
                drop_file.write(str(sent).strip() + '\n')

        print('TOTAL={}, ZEROCLASS={}'.format(i, zeros))

        """
        Statistics
622j/811
        texts.txt
        4716/6766 marked as zero with ('ORG', 'PRODUCT', 'FAC', 'PERSON', 'NORP', 'EVENT', 'DATE', 'MONEY')
        5515/6766 marked as zero with ('ORG', 'PRODUCT', 'FAC', 'NORP', 'EVENT')

        articles from database (summaries)
            without similarity test
            1144/4860 marked as zero with ('ORG', 'PRODUCT', 'FAC', 'NORP', 'EVENT') with original Glove
            1426/4860 marked as zero with ('ORG', 'PRODUCT', 'FAC', 'NORP', 'EVENT') with full 840B Glove

            with similarity test (0.6) with ('released', 'launched', 'updated', 'unveiled', 'new version', 'started')
            890/4860 marked as zero with ('ORG', 'PRODUCT', 'FAC', 'NORP', 'EVENT') with original Glove
            917/4860 marked as zero with ('ORG', 'PRODUCT', 'FAC', 'NORP', 'EVENT') with full 840B Glove

        """

    except KeyboardInterrupt:
        log.info('Keyboard Interrupt happened. Exiting. Last unhandled is #{}'.format(i))
        exit(0)


def test_articles():
    downloaders = [
        #blogs_parser.BlogsDownloader,
        #developerandeconomics.DevAndEconomicsDownloader,
        infoworld.InfoworldDownloader,
        #itnews.ItNewsDownloader,
        slashdot.SlashdotDownloader,
    ]
    nlp = English()
    log.info('Loaded spacy')
    raw_tags = (0, 1)
    tags = CategoricalTags(raw_tags)

    # testing article fetching
    # data_fetcher = ArticleTextFetch(downloaders=downloaders)
    # preprocessor = PreprocessTexts(nlp)

    # testing OpenIE
    data_fetcher = FileLineFetcher('../samples-json-relations.txt')
    preprocessor = PreprocessJsonOpenIEExtractions(nlp, nb_most_confident=1)
    wrapper = EventToSpanWrapper(nlp)



    # encoder = PaddingEncoder(nlp, tags, 30)
    encoder = Encoder(nlp, tags)
    tagger1 = HeuristicTagger(tags, nlp)
    tagger2 = DummyTagger(tags)
    tagger = ChainTagger(tags)
    tagger.add_tagger(tagger1)
    tagger.add_tagger(tagger2)

    try:
        ### Testing all pipeline:
        # neat usage
        # data_generator = encoder(wrapper(tagger(preprocessor(data_fetcher.get()))))
        # for x, y, sw in data_generator:
        #     print(np.shape(x), y, sw)

        # another example of usage
        # for texts in data_fetcher.test_get():
        #     for sent in preprocessor.sents(texts):
        #         for sent, tag in tagger.tag(sent):
        #             for x, y, sw in encoder((sent, tag)):
        #                 print(np.shape(x), y, sw)

        ### Testing tagger
        data_generator = tagger(preprocessor(data_fetcher.get_raw_lines()))
        for event, tag in data_generator:
            print(tag, event.sentence)


    except KeyboardInterrupt:
        log.info('Keyboard Interrupt happened. Exiting.')
        exit(0)


if __name__ == '__main__':
    log.basicConfig(filename='marking.log', format='%(levelname)s:%(message)s', level=log.DEBUG)
    test_tagger_texts()