import pickle
from collections import Counter
from experiments.marking.encoder import Encoder
from experiments.marking.tags import Tags


class LetterNGramEncoder(Encoder):
    def __init__(self, nlp, tags: Tags, corpora=None, vector_length=-1, ngram=3, dummy_char='^',
                 path_to_saved_vocab=None):
        """Instantiate encoder using raw text corpora or serialised vocab.

        Args:
            corpora: iterable of raw texts (str) for building vocabulary
            path_to_saved_vocab: load serialised vocabulary instead of processing corpora
            vector_length: length of the encodings (i.e. result vectors)
            ngram: use that number of symbols in ngram (default 3)
            dummy_char: symbol to use for extending tokens (e.g. when dummy_char='#': 'word' becames '#word#')
        """

        super().__init__(nlp, tags)

        # additional features about uppercase and wordvectors (see 'encode_token' method)
        self.nb_other_features = self.nlp.vocab.vectors_length + 5
        self.ngram = ngram
        self.dummy_char = dummy_char

        if path_to_saved_vocab:
            self.load_vocab(path_to_saved_vocab, vector_length)
        elif corpora:
            self.train(corpora, vector_length)

    def encode(self, text, tags):
        sent_enc = self.encode_text(text)
        tags_enc = self.encode_tags(tags)
        return sent_enc, tags_enc
        # return np.array(sent_enc), np.array(tags_enc)

    def encode_tags(self, raw_tags):
        return [self.tags.encode(raw_tag) for raw_tag in raw_tags]

    def decode_tags(self, tags_encoded):
        return [self.tags.decode(tag_enc) for tag_enc in tags_encoded]

    def encode_text(self, text):
        return [self.encode_token(token) for token in text]
        # return [self.nlp(str(token)).vector for token in text]
        # todo: remove that!

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
        t_spacied = self.nlp(t, tag=False, entity=False, parse=False)
        encoded.extend(t_spacied.vector)

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
        raw_vocab = Counter()
        for text in corpora:
            doc = self.nlp(text, tag=False, entity=False, parse=False)
            for i, token in enumerate(doc):
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


if __name__ == '__main__':
    # testing tag encoding and decoding
    from spacy.en import English
    from experiments.marking.tags import CategoricalTags
    raw_tags = ('O', 'I', 'B')
    encoder = LetterNGramEncoder(English(), CategoricalTags(raw_tags))

    truth = 'B O B O O O B B I O B I I I O'.split()
    print('Original:', truth)
    enc = encoder.encode_tags(truth)
    dec = encoder.decode_tags(enc)
    print(' Decoded:', dec)
    print(' Encoded:', enc)
    assert all(x == y for x, y in zip(truth, dec))
