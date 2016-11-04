import logging as log
from spacy.tokens import Span


class PreprocessTexts:
    def __init__(self, nlp):
        self.nlp = nlp

    def __call__(self, texts):
        # log.debug('Preprocessor: processing texts: {}'.format(texts))
        for text in texts:
            for sent in self.sents(text):
                yield sent

    def sents(self, text: str) -> Span:
        log.debug('Preprocessor: processing text: {}'.format(text))
        for sent in self.nlp(text).sents:
            if type(sent) is Span:
                log.debug('Preprocessor: yielding sent: {}'.format(text))
                yield sent


