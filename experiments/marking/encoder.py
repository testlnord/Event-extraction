import logging as log
from spacy.tokens import Span, Token
from experiments.marking.tags import Tags


class Encoder:
    def __init__(self, nlp, tags: Tags, none_tag_handler=None):
        """none_tag_handler allows us to add arbitrary logic for processing untagged objects.
        E.g. write them to file for further analysis."""
        def default_handler(text): pass
        if not none_tag_handler:
            self.none_tag_handler = default_handler
        self.nlp = nlp
        self.tags = tags

    def __call__(self, text_tag_pairs):
        """Accepts iterable of pairs (text: Span, raw_tag)
         and transforms them in the form suitable for machine learning algorithms"""
        for text, tag in text_tag_pairs:
            # if tag is None then ignore that pair
            if tag:
                yield self.encode(text, tag)
            else:
                self.none_tag_handler(text)

    def encode(self, text, tag):
        raise NotImplementedError


class SentenceEncoder(Encoder):
    def encode(self, text: Span, raw_tag):
        text_encoded = self.encode_text(text)
        tag_encoded = self.encode_tag(raw_tag)
        return text_encoded, tag_encoded

    def encode_text(self, text: Span):
        # text_encoded = [self._check_and_get_vector(tok) for tok in text]
        text_encoded = [tok.vector for tok in text]
        return text_encoded

    def encode_tag(self, raw_tag):
        return self.tags.encode(raw_tag)

    def _check_and_get_vector(self, token):
        if not (token.like_num or token.is_punct or token.is_space or token.has_vector):
            log.info('Encoder: word "{}" has no embedding (all zeros)!'.format(str(token)))
        return token.vector
