import logging as log
from spacy.tokens import Span
from experiments.marking.tags import Tags
from experiments.marking.abstract_encoder import Encoder


class SentenceEncoder(Encoder):
    def __init__(self, tags: Tags):
        self.tags = tags

    def encode_data(self, text: Span):
        text_encoded = [tok.vector for tok in text]
        return text_encoded

    def encode_tags(self, raw_tag):
        return self.tags.encode(raw_tag)

    @property
    def nbclasses(self):
        return len(self.tags)
