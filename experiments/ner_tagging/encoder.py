import logging as log
import os
import pickle
from collections import Counter
from spacy.tokens import Span
from experiments.marking.tags import Tags, CategoricalTags
from experiments.marking.abstract_encoder import Encoder


class LetterNGramEncoder(Encoder):
    def __init__(self, tags: Tags, ngram=3, word_vectors_length=300):
        """
        :param tags: tags for encoding
        :param ngram: use that number of symbols in ngram (default 3)
        """
        # Additional features about uppercase and wordvectors (see 'encode_token' method)
        self.tags = tags
        self.nb_other_features = word_vectors_length + 5
        self.ngram = ngram
        # Use some rare symbol - don't change that to keep consistency with old trained vocabs
        self.dummy_char = '^'

    @classmethod
    def from_vocab_file(cls, tags: Tags, vocab_path=None, force_vector_length=-1, word_vectors_length=300):
        # Determine path of the vocabulary
        vocab_dir = os.path.join(os.path.dirname(__file__), '../models')
        if not vocab_path:
            vocab_name = 'encoder_vocab_default.bin'
            vocab_path = os.path.join(vocab_dir, vocab_name)

        # Construct encoder and load vocabulary
        if vocab_path and os.path.isfile(vocab_path):
            encoder = cls(tags, word_vectors_length=word_vectors_length)
            encoder.load_vocab(vocab_path, force_vector_length)
            log.info('LetterNGramEncoder: Loaded vocabulary. Vector length is {}'.format(encoder.vector_length))
            return encoder

        raise ('LetterNGramEncoder: vocab with path {} is not found. Cannot construct encoder'.format(vocab_path))

    def encode_tags(self, raw_tags):
        return [self.tags.encode(raw_tag) for raw_tag in raw_tags]

    def decode_tags(self, tags_encoded):
        return [self.tags.decode(tag_enc) for tag_enc in tags_encoded]

    def encode_data(self, text: Span):
        return [self.encode_token(token) for token in text]

    def encode_token(self, token):
        t = str(token)
        tl = t.lower()
        ngrams = list(self.ngrams(tl))

        known_ngrams = [ngram for ngram in ngrams if ngram in self.dvocab.keys()]
        unknown_there = 1 - len(known_ngrams) / len(ngrams)
        encoded = [0] * len(self.dvocab)
        indexes = [self.dvocab[ngram] for ngram in known_ngrams]
        for index in indexes:
            encoded[index] = 1

        # adding other features

        # adding word embedding - pretty useful information
        encoded.extend(token.vector)

        # preserving information about uppercase
        upmask = [int(c.isupper()) for c in t]
        # avoiding errors because of too short tokens
        while len(upmask) < 2:
            upmask.append(0)
        additional_features = [unknown_there,
                               upmask[0],
                               int(any(upmask[1:])),
                               int(all(upmask)),
                               sum(upmask) / len(upmask),
                               ]
        encoded.extend(additional_features)

        return encoded

    def train(self, corpora, vector_length=-1):
        """
        Train encoder on corpora (i.e. collect most frequent ngrams from corpora)
        :param corpora: iterable of words for building vocabulary
        :param vector_length: length of the encodings (i.e. result vectors), if -1 then keep all ngrams
        :return: None
        """
        raw_vocab = Counter()
        for i, token in enumerate(corpora):
            t = token.text.lower()
            raw_vocab += Counter(self.ngrams(t))

        self.vocab = list(map(lambda item: item[0], raw_vocab.most_common()))
        self.set_vector_length(vector_length)
        self._make_dict_vocab()

    def save_vocab(self, dir='.'):
        """Save vocabulary and return path to it"""
        path = dir + '/encoder_vocab_{}gram_{}len.bin'.format(self.ngram, self.vector_length)
        if self.vocab is not None:
            with open(path, 'wb') as f:
                pickle.dump(self.vocab, f)
        return path

    def load_vocab(self, path, vector_length=-1):
        with open(path, 'rb') as f:
            self.vocab = pickle.load(f)
            self.ngram = len(self.vocab[0])
            self.set_vector_length(vector_length)
            self._make_dict_vocab()

    def ngrams(self, token):
        t = self.dummy_char + str(token) + self.dummy_char
        for j in range(len(t) - self.ngram + 1):
            ngram = t[j:j+self.ngram]
            yield ngram

    def _make_dict_vocab(self):
        self.dvocab = dict((item, i) for i, item in enumerate(self.vocab))

    @property
    def vector_length(self):
        return self._vector_length

    def set_vector_length(self, vector_length):
        # real vector_length = core vector_length + nb_other_features
        core_length = vector_length - self.nb_other_features
        vocab_length = len(self.vocab)

        if core_length <= 0:
            # either provided vector_length is too small or is default value
            self._vector_length = vocab_length + self.nb_other_features
        elif core_length < vocab_length:
            # cut vocab, throwing least frequent entries
            self.vocab = self.vocab[:core_length]
            self._vector_length = vector_length
        else:
            # just add useless ngrams to the vocab to keep vector length as desired
            nb_pad_values = vocab_length - core_length
            pad_value = self.dummy_char * self.ngram
            self.vocab.extend([pad_value] * nb_pad_values)
            self._vector_length = vector_length
