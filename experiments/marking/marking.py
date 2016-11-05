import logging as log
import numpy as np
from spacy.en import English
from spacy.tokens import Span

from db.db_handler import DatabaseHandler
from data_mining.downloaders import article_downloaders
from data_mining.downloaders import blogs_parser, developerandeconomics, infoworld, itnews, slashdot

from experiments.marking.data_fetcher import *
from experiments.marking.preprocessor import *
from experiments.marking.tagger import *
from experiments.marking.encoder import *


class EventToSpanWrapper:
    def __init__(self, nlp):
        self.nlp = nlp

    def __call__(self, event_tag_pairs):
        for event, tag in event_tag_pairs:
            yield self.nlp(event.sentence)[:], tag

def test():
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
    tagger1 = HeuristicEventTagger(tags, nlp)
    tagger2 = DummyTagger(tags)
    tagger = ChainTagger(tags)
    tagger.add_tagger(tagger1)
    tagger.add_tagger(tagger2)

    try:
        # neat usage
        data_generator = encoder(wrapper(tagger(preprocessor(data_fetcher.get()))))
        for x, y, sw in data_generator:
            print(np.shape(x), y, sw)

        # another example of usage
        # for texts in data_fetcher.test_get():
        #     for sent in preprocessor.sents(texts):
        #         for sent, tag in tagger.tag(sent):
        #             for x, y, sw in encoder((sent, tag)):
        #                 print(np.shape(x), y, sw)
    except KeyboardInterrupt:
        log.info('Keyboard Interrupt happened. Exiting.')
        exit(0)


if __name__ == '__main__':
    log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)
    test()