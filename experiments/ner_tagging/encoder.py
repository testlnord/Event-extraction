from collections import Counter
import numpy as np
from experiments.marking.encoder import Encoder


class LetterNGramEncoder(Encoder):
    def __init__(self, nlp, tags, corpora, vector_length=-1, ngram=3, dummy_char='^'):
        super().__init__(nlp, tags)

        self.nb_other_features = 4 + 1 # see 'encode_token' method
        self.vector_length = vector_length
        self.ngram = ngram
        self.dummy_char = dummy_char
        self.train(corpora)

    def encode(self, text, tags):
        sent_enc = self.encode_text(text)
        tags_enc = self.encode_tags(tags)
        return np.array(sent_enc), np.array(tags_enc)

    def encode_tags(self, raw_tags):
        return [self.tags.encode(raw_tag) for raw_tag in raw_tags]

    def encode_text(self, text):
        return [self.encode_token(token) for token in text]

    def encode_token(self, token):
        t = str(token)
        tl = t.lower()
        ngrams = list(self._ngrams(tl))
        preencoded = [int(entry in ngrams) for entry in self.vocab]

        # all unknown ngrams encoded as one class (last bit in the vector of size = len(self.vocab) + 1)
        # so, the problem of possible missed (unseen) ngrams addressed here
        unknown_there = any(ngram not in self.vocab for ngram in ngrams)
        preencoded.append(int(unknown_there))

        # preserving information about uppercase
        upmask = [int(c.isupper()) for c in t]
        while len(upmask) < 2:
            upmask.append(0)
        additional_encodings = [upmask[0], int(any(upmask[1:])), int(all(upmask)), sum(upmask) / len(upmask)]

        encoded = additional_encodings + preencoded
        return encoded

    def train(self, corpora):
        raw_vocab = Counter()
        for text in corpora:
            doc = self.nlp(text, tag=False, entity=False, parse=False)
            for i, token in enumerate(doc):
                t = token.text.lower()
                raw_vocab += Counter(self._ngrams(t))

        top_n = None if self.vector_length < 1 else self.vector_length - self.nb_other_features
        self.vocab = list(map(lambda item: item[0], raw_vocab.most_common(top_n)))
        self.vector_length = len(self.vocab) + self.nb_other_features

    def _ngrams(self, token):
        t = self.dummy_char + str(token) + self.dummy_char
        for j in range(len(t) - self.ngram + 1):
            ngram = t[j:j+self.ngram]
            yield ngram
