import logging as log
import numpy as np
from spacy.tokens import Span
from experiments.marking.tagger import Tags


class Encoder:
    def __init__(self, nlp, tags: Tags):
        self.nlp = nlp
        self.tags = tags

    def __call__(self, text_tag_pairs):
        for text, tag in text_tag_pairs:
            # if tag is None then ignore
            if tag:
                yield self.encode(text, tag)

    def encode(self, text: Span, raw_tag) -> (np.ndarray, np.ndarray, np.ndarray):
        text_encoded, sample_weights = self.encode_text(text)
        tag_encoded = self.encode_tag(raw_tag)
        return np.array(text_encoded), np.array(tag_encoded), np.array(sample_weights)

    def encode_text(self, text: Span):
        text_encoded = [tok.vector for tok in text]
        sample_weights = [1] * len(text_encoded)
        return text_encoded, sample_weights

    def encode_tag(self, raw_tag):
        return self.tags.encode(raw_tag)


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


