import logging as log
from spacy.tokens import Span
from experiments.event_with_offsets import EventOffsets


class Tags:
    def encode(self, raw_tag) -> list:
        raise NotImplementedError

    def decode(self, cat):
        raise NotImplementedError

    def is_correct(self, raw_or_cat) -> bool:
        raise NotImplementedError

    @property
    def default_tag(self):
        return None


class CategoricalTags(Tags):
    """Provided tags must allow convertation to string without loss of information"""
    def __init__(self, raw_tags):
        self.raw_tags = tuple(map(str, raw_tags))
        self.nbtags = len(raw_tags)

    def encode(self, raw_tag):
        return self._to_categorical(raw_tag)

    def decode(self, cat):
        return self._to_raw(cat)

    # todo: is it correct
    def is_correct(self, raw_or_cat):
        return raw_or_cat in self.raw_tags or self.encode(raw_or_cat) in self.raw_tags

    @property
    def default_tag(self):
        return '0'

    def _to_categorical(self, raw_tag):
        cat = [0] * self.nbtags
        cat[self.raw_tags.index(str(raw_tag))] = 1
        return cat

    def _to_raw(self, cat) -> str:
        i = cat.index(1)
        return self.raw_tags[i]


class Tagger:
    """Common Tagger interface. Taggers expected to return None when they can't assign tag."""
    def __init__(self, tags: Tags):
        self.tags = tags

    def __call__(self, objs):
        for obj in objs:
            yield obj, self.tag(obj)

    def tag(self, obj):
        raise NotImplementedError


class ChainTagger(Tagger):
    """Applies provided taggers until first success."""
    def __init__(self, tags):
        self.taggers = []
        super().__init__(tags)

    def tag(self, text):
        for tagger in self.taggers:
            res = tagger.tag(text)
            if res:
                return res
        return None

    def add_tagger(self, tagger):
        """Add tagger to the end of the chain"""
        if tagger.tags != self.tags:
            raise ValueError("Please provide tagger with the same tags!")
        self.taggers.append(tagger)


class DummyTagger(Tagger):
    def tag(self, text):
        # return self.tags.default_tag
        return '1'


class TextUserTagger(Tagger):
    def __init__(self, tags: Tags, escape_input=''):
        self.end = escape_input
        super(TextUserTagger, self).__init__(tags)

    def tag(self, text):
        str_text= str(text).strip().replace('\n', ' ')
        output_sep = '_' * len(str_text) + '\n'
        prompt = output_sep + str_text + '\n' + 'enter tag: '

        user_input = None
        while not self.tags.is_correct(user_input):
            user_input = input(prompt)
            if user_input == self.end:
                break
        return user_input


class HeuristicEventTagger(Tagger):
    def __init__(self, tags, nlp,
                 keyphrases=('released', 'started', 'updated', 'unveiled'),
                 min_similarity=0.4,
                 suspicious_ne_types=('ORG', 'PRODUCT', 'FAC', 'PERSON', 'NORP', 'EVENT', 'DATE', 'MONEY')):
        self.nlp = nlp
        self.keyphrases = map(nlp, keyphrases)
        self.min_similarity = min_similarity
        self.ne_types = suspicious_ne_types

        super().__init__(tags)

    def tag(self, event: EventOffsets):
        # we want to have all extracted tokens to be related to one doc
        # only that way we can investigate connections (e.g. dependency tree links) between them
        doc = self.nlp(event.sentence)
        ent1_tokens = self._extract_tokens(event.entity1_offsets, doc)
        action_tokens = self._extract_tokens(event.action_offsets, doc)
        ent2_tokens = self._extract_tokens(event.entity2_offsets, doc)

        log.debug('HeuristicTagger: ent1_tokens={}'.format(ent1_tokens))
        log.debug('HeuristicTagger: action_tokens={}'.format(action_tokens))
        log.debug('HeuristicTagger: ent2_tokens={}'.format(ent2_tokens))

        # now is the time to use heuristics; we apply:
        # - named entity type matcher to entity1 and entity2 (there unlikely something interesting in action)
        # - keyword matcher for all tokens in event
        # - matcher of the unknown words: words that has no word embedding
        res = self._ner_match(ent1_tokens + ent2_tokens) or \
              self._unknown_words_matcher(ent1_tokens + ent2_tokens) or \
              self._similar_words_match(action_tokens + ent1_tokens + ent2_tokens)

        # that tagger is careful: it gives up when finds something
        # only when it finds nothing, it assigns default tag (e.g. '0' - category of negative examples)
        _tag = None
        if not res:
            _tag = self.tags.default_tag
            log.info('HeuristicTagger: tagged event "{}" with tag={}'.format(repr(event), _tag))
        return _tag
        # return None if res else self.tags.default_tag

    def _ner_match(self, tokens):
        """Returns True if any of the tokens has Named Entity type from provided in constructor list of these types"""
        return any(tok.ent_type_ in self.ne_types for tok in tokens)

    def _similar_words_match(self, tokens):
        """Returns True if any of the tokens is similar enough to any of the keywords (provided in constructor)"""
        return any(any(tok.similarity(keyphrase) > self.min_similarity for keyphrase in self.keyphrases)
                   for tok in tokens)

    # todo: ignor stupid numbers, tokens of few symbols and the like
    def _unknown_words_matcher(self, tokens):
        """Very suspicious situation!"""
        return any(not tok.has_vector for tok in tokens)

    def _extract_tokens(self, offsets, doc):
        """Extract tokens from doc which fall in specified offsets"""
        return [tok for tok in doc if any((offset[0] <= tok.idx < offset[1]) for offset in offsets)]


