import logging as log
import re
import os

from db.db_handler import DatabaseHandler
from data_mining.downloaders import article_downloaders


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

    # todo:
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
                    yield text

    def get_old(self):
        """Yield articles from database"""
        for article in self.db.get_articles():
            for text in self._texts_from_article(article):
                yield text

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


class FilesFetcher:
    @staticmethod
    def get_texts_from_files(paths):
        """Get texts from files specified in paths"""
        for path in paths:
            try:
                with open(path) as f:
                    log.info('FilesFetcher: reading file {}'.format(path))
                    yield f.read()

            except Exception as e:
                log.warning('FilesFetcher: exception {} while opening path {}'.format(e.args, repr(path)))

    @staticmethod
    def get_texts_from_filetree(root_dir, pattern='.+\.txt$'):
        """Get texts from all files in the filesystem tree starting in root_dir, that match regex pattern"""
        paths = FilesFetcher.get_paths(root_dir, pattern)
        for text in FilesFetcher.get_texts_from_files(paths):
            yield text

    @staticmethod
    def get_paths(root_dir, pattern='.+\.txt$'):
        matcher = re.compile(pattern)
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for fname in filenames:
                if matcher.match(fname) and fname[0] != '.':
                    filepath = os.path.join(dirpath, fname)
                    yield filepath


class FileLineFetcher:
    def __init__(self, path, nlp):
        self.path = path
        self.nlp = nlp

    def get_raw_lines(self):
        with open(self.path) as f:
            for line in f.readlines():
                yield line

    def get_with_tags(self):
        with open(self.path) as f:
            for line in f.readlines():
                sent_and_tag = line.split(';')
                raw_sent = sent_and_tag[0]
                raw_tag = sent_and_tag[1]
                yield self.nlp(raw_sent)[:], str.strip(raw_tag)
