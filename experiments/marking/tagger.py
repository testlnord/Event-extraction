import logging as log
from spacy.tokens import Span
from experiments.event_with_offsets import EventOffsets
from experiments.marking.tags import Tags


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
    def __init__(self, tags, dummy_tag=None):
        super().__init__(tags)

        self.dummy_tag = dummy_tag
        # self.dummy_tag = self.tags.default_tag
        # if dummy_tag and self.tags.is_correct(dummy_tag):
        # else:
        #     log.warning('DummyTagger: provided dummy tag is not correct! Using default tag={}'.format(self.dummy_tag))

    def tag(self, text):
        return self.dummy_tag


#todo: suitable only for spans
class TextUserTagger(Tagger):
    def __init__(self, tags: Tags, escape_input=''):
        self.end = escape_input
        super().__init__(tags)

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


class HeuristicTagger(Tagger):
    def __init__(self, tags, nlp,
                 keyphrases=('released', 'started', 'updated', 'unveiled'),
                 min_similarity=0.4,
                 suspicious_ne_types=('ORG', 'PRODUCT', 'FAC', 'PERSON', 'NORP', 'EVENT', 'DATE', 'MONEY')):
        self.nlp = nlp
        self.keyphrases = map(nlp, keyphrases)
        self.min_similarity = min_similarity
        self.ne_types = suspicious_ne_types

        super().__init__(tags)

    def tag(self, obj):
        raise NotImplementedError

    def _ner_match(self, tokens):
        """Returns True if any of the tokens has Named Entity type from provided in constructor list of these types"""
        # return any(tok.ent_type_ in self.ne_types for tok in tokens)

        for tok in tokens:
            if tok.ent_type_ in self.ne_types:
                log.debug('HeuristicTagger: _ner_match: token {}; tok.ent_type "{}"'.format(repr(tok.text), tok.ent_type_))
                return True
        return False

    def _similar_words_match(self, tokens):
        """Returns True if any of the tokens is similar enough to any of the keywords (provided in constructor)"""
        # return any(any(tok.similarity(keyphrase) > self.min_similarity for keyphrase in self.keyphrases) for tok in tokens)

        for tok in tokens:
            for keyphrase in self.keyphrases:
                if tok.similarity(keyphrase) > self.min_similarity:
                    log.debug('HeuristicTagger: _similar_words_match: token {}; keyphrase {}'.format(repr(tok.text), repr(keyphrase)))
                    return True
        return False

    def _unknown_words_matcher(self, tokens):
        """True if there're words without word embedding: very suspicious situation! (e.g. JetBrains)"""
        # return any(not (tok.like_num or tok.is_punct or tok.is_space or tok.has_vector) for tok in tokens)

        for tok in tokens:
            if not (tok.like_num or tok.is_punct or tok.is_space or tok.has_vector):
                log.debug('HeuristicTagger: _unknown_words_match: token {}'.format(repr(tok.text)))
                return True
        return False


class HeuristicSpanTagger(HeuristicTagger):
    def tag(self, span: Span):
        res = self._ner_match(span) or \
              self._unknown_words_matcher(span) or \
              self._similar_words_match(span)

        _tag = None
        if not res:
            _tag = self.tags.default_tag
            log.debug('HeuristicTagger: tagged text {}; tag={}'.format(repr(span.text), _tag))
        return _tag


class HeuristicEventTagger(HeuristicTagger):
    def tag(self, event: EventOffsets):
        # we want to have all extracted tokens to be related to one doc
        # only that way we can investigate connections (e.g. dependency tree links) between them
        doc = self.nlp(event.sentence)
        ent1_tokens = self._extract_tokens(event.entity1_offsets, doc)
        action_tokens = self._extract_tokens(event.action_offsets, doc)
        ent2_tokens = self._extract_tokens(event.entity2_offsets, doc)

        # just to check if implementation of extraction is correct
        # log.debug('HeuristicTagger: ent1_tokens={}'.format(ent1_tokens))
        # log.debug('HeuristicTagger: action_tokens={}'.format(action_tokens))
        # log.debug('HeuristicTagger: ent2_tokens={}'.format(ent2_tokens))

        # now is the time to use heuristics; we apply:
        # - named entity type matcher to entity1 and entity2 (there unlikely something interesting in action)
        # - keyword matcher for all tokens in event
        # - matcher of the unknown words: words that has no word embedding
        res = self._ner_match(ent1_tokens + ent2_tokens) or \
              self._unknown_words_matcher(ent1_tokens + ent2_tokens) or \
              self._similar_words_match(action_tokens + ent1_tokens + ent2_tokens)

        # that tagger is careful: it gives up when finds something
        # only when it finds nothing, it assigns default tag (e.g. '0' - category of negative examples)
        # return None if res else self.tags.default_tag
        _tag = None
        if not res:
            _tag = self.tags.default_tag
            log.debug('HeuristicTagger: tagged event {}; tag={}'.format(repr(event), _tag))
        return _tag

    def _extract_tokens(self, offsets, doc):
        """Extract tokens from doc which fall in specified offsets.
        In other words, map offsets to actual tokens."""
        return [tok for tok in doc if any((offset[0] <= tok.idx < offset[1]) for offset in offsets)]
