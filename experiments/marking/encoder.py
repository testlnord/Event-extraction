import logging as log
from spacy.tokens import Span
from experiments.marking.tags import Tags
from experiments.marking.abstract_encoder import Encoder


class SentenceEncoder(Encoder):
    def __init__(self, tags: Tags):
        self.tags = tags

    def encode_data(self, text: Span):
        # text_encoded = [self._check_and_get_vector(tok) for tok in text]
        text_encoded = [tok.vector for tok in text]
        return text_encoded

    def encode_tag(self, raw_tag):
        return self.tags.encode(raw_tag)

    def _check_and_get_vector(self, token):
        if not (token.like_num or token.is_punct or token.is_space or token.has_vector):
            log.info('Encoder: word "{}" has no embedding (all zeros)!'.format(str(token)))
        return token.vector
