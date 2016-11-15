from collections import Counter

class LetterNGramEncoder:
    def __init__(self, nlp, corpora, vector_length=-1, ngram=3, dummy_char='^'):
        self.nlp = nlp
        num_additional_features = 4 + 1 # see 'encode' method
        self.vector_length = vector_length - num_additional_features
        self.ngram = ngram
        self.dummy_char = dummy_char
        self.train(corpora)

    def __call__(self, tokens):
        for token in tokens:
            yield self.encode(token)

    def encode(self, token):
        t = str(token)
        tl = t.lower()
        ngrams = list(self._ngrams(tl))
        preencoded = [int(entry in ngrams) for entry in self.vocab]

        # all unknown ngrams encoded as one class (last bit in the vector of size = len(self.vocab) + 1)
        # so, the problem of possible missed (unseen) ngrams addressed here
        unknown_there = any(ngram not in self.vocab for ngram in ngrams)
        preencoded.append(int(unknown_there))

        # preserving information about uppercase
        upmask = [int(c.isupper) for c in t]
        while len(upmask) < 2:
            upmask.append(0)
        additional_encodings = [upmask[0], int(any(upmask[1:])), int(all(upmask)), sum(upmask) / len(upmask)]

        encoded = additional_encodings + preencoded
        return encoded

    def train(self, corpora: str):
        raw_vocab = Counter()
        doc = self.nlp(corpora, tag=False, entity=False, parse=False)
        for i, token in enumerate(doc):
            t = token.text.lower()
            raw_vocab += Counter(self._ngrams(t))

        top_n = None if self.vector_length < 1 else self.vector_length
        self.vocab = list(map(lambda item: item[0], raw_vocab.most_common(top_n)))

    def _ngrams(self, token):
        t = self.dummy_char + str(token) + self.dummy_char
        for j in range(len(t) - self.ngram + 1):
            l = self.ngram * j
            ngram = t[l:l+self.ngram]
            yield ngram

