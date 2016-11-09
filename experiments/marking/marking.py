import logging as log
import numpy as np
from spacy.en import English
from spacy.tokens import Span

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

def test_tagger_texts():
    nlp = English()
    log.info('Loaded spacy')
    raw_tags = (0, 1)
    tags = CategoricalTags(raw_tags)

    drop_file = open('../untagged_last.txt', 'a')
    data1 = FilesFetcher.get_texts_from_files(['../texts.txt'])
    data2 = FilesFetcher.get_texts_from_filetree('/media/Documents/datasets/OANC-GrAF')
    preprocessor = PreprocessTexts(nlp, min_words_in_sentence=3)
    # encoder = PaddingEncoder(nlp, tags, 50)

    suspicious_ne_types=('ORG', 'PRODUCT', 'FAC', 'NORP', 'EVENT')
    tagger1 = HeuristicSpanTagger(tags, nlp, suspicious_ne_types=suspicious_ne_types)
    tagger2 = DummyTagger(tags, None)
    tagger = ChainTagger(tags)
    tagger.add_tagger(tagger1)
    tagger.add_tagger(tagger2)

    try:
        data_generator = tagger(preprocessor(data2))
        zeros = 0
        total = 0
        for total, (sent, tag) in enumerate(data_generator):
            # print(i, ':', tag, sent)
            if str(tag) == '0': zeros += 1

            if not tag:
                # write to file for further analysis
                drop_file.write(str(sent).strip() + '\n')

        print('TOTAL={}, ZEROCLASS={}'.format(total+1, zeros))

        """
        Statistics with Sherlock as input:
        4716/6766 marked automatically with ('ORG', 'PRODUCT', 'FAC', 'PERSON', 'NORP', 'EVENT', 'DATE', 'MONEY')
        5515/6766 marked automatically with ('ORG', 'PRODUCT', 'FAC', 'NORP', 'EVENT')
        """

    except KeyboardInterrupt:
        log.info('Keyboard Interrupt happened. Exiting.')
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
    log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)
    fh = log.FileHandler('marking.log')
    test_tagger_texts()