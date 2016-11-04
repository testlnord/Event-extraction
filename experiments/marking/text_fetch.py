import logging as log
from itertools import islice
import numpy as np
from spacy.en import English
from spacy.tokens import Span

from db.db_handler import DatabaseHandler
from data_mining.downloaders import article_downloaders
from data_mining.downloaders import blogs_parser, developerandeconomics, infoworld, itnews, slashdot

from experiments.marking.preprocessor import *
from experiments.marking.tagger import *
from experiments.marking.encoder import *


# todo: handle incorrect downloader behaviour
class ArticleTextFetch:
    """Wrapper for article downloaders. Handles errors, incorrect downloaders behaviour,
    saves state of article downloading."""
    def __init__(self, return_header=False, return_summary=True, return_article_text=False,
                 downloaders=article_downloaders):
        self.header = return_header
        self.summary = return_summary
        self.text = return_article_text
        self.downloaders = downloaders
        self.db = DatabaseHandler()

    def get(self, only_new=True):
        """Download articles, add them to database and yield if they are not already in db """
        # article_generators = map(lambda d: d.get_articles(), self.downloaders)

        # todo:
        max_nb_subsequent_falls = 10
        for downloader in self.downloaders:
            falls = 0
            fall = False
            # starting timer in async mode
            # if timer ends: falls += 1
            if falls >= max_nb_subsequent_falls:
                continue

            for article in downloader.get_articles():
                # resetting timer
                res = self.db.add_article_or_get_id(article)
                log.debug('Fetcher: add_article_or_get_id() returned {}'.format(res))
                if only_new and False:
                    continue

                yield self._texts_from_article(article)

    def get_old(self):
        """Yield articles from database"""
        for article in self.db.get_articles():
            yield self._texts_from_article(article)

    def test_get_async(self):
        """Dummy testing method"""
        import asyncio
        def gen(d, n):
            """Generators of n integer numbers kratnih d"""
            for i in range(1, n + 1):
                yield i * d
        n = 3
        generators = [gen(i, n) for i in (2,3,5,7)]

        for gen in generators:
            for i in gen:
                yield i


    def test_get(self):
        """Dummy testing method"""
        for i in range(10):
            t = 'This is a simple test dummy sentence number {}'.format(i)
            log.debug('Fetcher: yielding: {}'.format(t))
            yield t

    def _texts_from_article(self, article):
        t = []
        if self.header:
            t.append(article.header)
        elif self.summary:
            t.append(article.summary)
        elif self.text:
            t.append(article.text)

        if not t:
            log.warning('ArticleTextFetch: no output specified! Returning empty tuple.')

        return tuple(t)



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

    data_fetcher = ArticleTextFetch(downloaders=downloaders)
    preprocessor = PreprocessTexts(nlp)
    tags = CategoricalTags(raw_tags)
    tagger = DummyTagger(tags)
    encoder = PaddingEncoder(nlp, tags, 30)

    # neat usage
    data_generator = encoder(tagger(preprocessor(data_fetcher.test_get())))
    for x, y, sw in data_generator:
        print(np.shape(x), y, sw)

    # another example of usage
    # for texts in data_fetcher.test_get():
    #     for sent in preprocessor.sents(texts):
    #         for sent, tag in tagger.tag(sent):
    #             for x, y, sw in encoder((sent, tag)):
    #                 print(np.shape(x), y, sw)

if __name__ == '__main__':
    log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)
    test()