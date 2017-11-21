import logging as log
import re
import os

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

    def get_with_tags(self, separator=';'):
        with open(self.path) as f:
            for line in f.readlines():
                sent_and_tag = line.split(sep=separator)
                raw_sent = sent_and_tag[0]
                raw_tag = sent_and_tag[1]
                yield self.nlp(raw_sent)[:], str.strip(raw_tag)
