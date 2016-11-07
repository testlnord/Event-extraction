import logging as log
import numpy as np
from spacy.tokens import Span, Token
from experiments.marking.tags import Tags


class Encoder:
    def __init__(self, nlp, tags: Tags):
        self.nlp = nlp
        self.tags = tags

    def __call__(self, text_tag_pairs):
        """Accepts iterable of pairs (text: Span, raw_tag)
         and transforms them in the form suitable for machine learning algorithms"""
        for text, tag in text_tag_pairs:
            # if tag is None then ignore that pair
            if tag:
                yield self.encode(text, tag)

    def encode(self, text: Span, raw_tag) -> (np.ndarray, np.ndarray, np.ndarray):
        """Encodes single pair of (text: Span, raw_tag).
        Also returns mask, representing actual values (non-zero) in returned encoded text."""
        text_encoded, sample_weights = self.encode_text(text)
        tag_encoded = self.encode_tag(raw_tag)
        return np.array(text_encoded), np.array(tag_encoded), np.array(sample_weights)

    def encode_text(self, text: Span):
        # text_encoded = [self._check_and_get_vector(tok) for tok in text]
        text_encoded = [tok.vector for tok in text]
        sample_weights = [1] * len(text_encoded)
        return text_encoded, sample_weights

    def encode_tag(self, raw_tag):
        return self.tags.encode(raw_tag)

    def _check_and_get_vector(self, token):
        if not (token.like_num or token.is_punct or token.is_space or token.has_vector):
            log.info('Encoder: word "{}" has no embedding (all zeros)!'.format(str(token)))
        return token.vector


class PaddingEncoder(Encoder):
    def __init__(self, nlp, tags, pad_to_length, pad_value=0, pad_tags=False):
        self.pad_to_length = pad_to_length
        self.pad_value = pad_value
        self.pad_tags = pad_tags
        super(PaddingEncoder, self).__init__(nlp, tags)

    def encode(self, text, raw_tag):
        text_encoded, sample_weights = self.encode_text(text)
        tag_encoded = self.encode_tag(raw_tag)

        ones = len(text_encoded)
        zeros = self.pad_to_length - ones
        pad_text = ((0, zeros), (0, 0))

        # mask, telling us about inserted redundant values
        sample_weights = np.array([1] * ones + [0] * zeros)
        text_encoded = np.pad(text_encoded, pad_width=pad_text, mode='constant', constant_values=self.pad_value)
        if self.pad_tags:
            tag_encoded = np.pad(tag_encoded, pad_width=pad_text, mode='constant', constant_values=self.pad_value)

        return text_encoded, tag_encoded, sample_weights


