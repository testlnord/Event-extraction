import logging as log
from db.db_handler import DatabaseHandler
from data_mining.downloaders import article_downloaders


# todo: handle incorrect downloader behaviour
class ArticleTextFetcher:
    """Wrapper for article downloaders. Handles errors, incorrect downloaders behaviour,
    saves state of article downloading."""
    def __init__(self, return_header=False, return_summary=True, return_article_text=False,
                 downloaders=article_downloaders):
        self.header = return_header
        self.summary = return_summary
        self.text = return_article_text
        self.downloaders = downloaders
        self.db = DatabaseHandler()

    # todo: get new articles
    def get(self, only_new=True):
        """Download articles, add them to database and yield if they are not already in db """
        # article_generators = map(lambda d: d.get_articles(), self.downloaders)

        for downloader in self.downloaders:
            for article in downloader.get_articles():
                res = self.db.add_article_or_get_id(article)
                log.debug('Fetcher: add_article_or_get_id() returned {}'.format(res))
                if only_new:
                    continue

                for text in self._texts_from_article(article):
                    yield article, text

    def get_old(self):
        """Yield articles from database"""
        for article in self.db.get_articles():
            for text in self._texts_from_article(article):
                yield article, text

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
