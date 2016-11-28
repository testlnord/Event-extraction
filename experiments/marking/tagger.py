import logging as log
from spacy.tokens import Span
from experiments.event_with_offsets import EventOffsets
from experiments.marking.tags import Tags


class ChainTagger:
    """Applies taggers in sequence until first success (not None tag)."""
    def __init__(self, tags):
        self.taggers = []
        self.tags = tags

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


class DummyTagger:
    def __init__(self, tags, dummy_tag=None):
        self.tags = tags
        self.dummy_tag = dummy_tag
        # self.dummy_tag = self.tags.default_tag
        # if dummy_tag and self.tags.is_correct(dummy_tag):
        # else:
        #     log.warning('DummyTagger: provided dummy tag is not correct! Using default tag={}'.format(self.dummy_tag))

    def tag(self, text):
        return self.dummy_tag


class TextUserTagger:
    def __init__(self, tags: Tags, escape_input=''):
        self.end = escape_input
        self.tags = tags

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


# todo: use dependency tree (or POS tags) in heuristics
class HeuristicTagger:
    def __init__(self, tags, nlp,
                 keyphrases=('released', 'launched', 'updated', 'unveiled',
                             # 'upgrade', 'announced', 'available', 'introduces', 'coming',
                             # 'changes', 'share', 'started', redesigned', 'revised', 'fixed', 'improved', 'designed',
                             # 'newest', 'latest', 'better', 'finally',
                             # 'enhancements', 'features', 'build', 'version',
                 ),
                 min_similarity=0.6,
                 suspicious_ne_types=('ORG', 'PRODUCT', 'FAC', 'PERSON', 'NORP', 'EVENT', 'DATE', 'MONEY')):
        self.tags = tags
        self.nlp = nlp
        self.keyphrases = [nlp(k) for k in keyphrases]
        self.min_similarity = min_similarity
        self.ne_types = suspicious_ne_types

    def _ner_match(self, tokens):
        """Returns True if any of the tokens has Named Entity type from provided in constructor list of these types"""
        # return any(tok.ent_type_ in self.ne_types for tok in tokens)

        for tok in tokens:
            if tok.ent_type_ in self.ne_types:
                log.debug('{}: _ner_match: token {}; tok.ent_type {}'.format(type(self).__name__, repr(tok.text), repr(tok.ent_type_)))
                return True
        return False

    def _similar_words_match(self, tokens):
        """Returns True if any of the tokens is similar enough to any of the keywords (provided in constructor)"""
        # return any(any(tok.similarity(keyphrase) > self.min_similarity for keyphrase in self.keyphrases) for tok in tokens)

        for tok in tokens:
            for keyphrase in self.keyphrases:
                ss = tok.similarity(keyphrase)
                if ss > self.min_similarity:
                    log.debug('{}: _similar_words_match: {:.2f}; token {}; keyphrase {}'.format(type(self).__name__, ss, repr(tok.text).ljust(12), repr(keyphrase.text)))
                    return True
        return False

    def _unknown_words_match(self, tokens):
        """True if there're words without word embedding: very suspicious situation! (e.g. JetBrains)"""
        # return any(not (tok.like_num or tok.is_punct or tok.is_space or tok.has_vector) for tok in tokens)

        for tok in tokens:
            if not (tok.like_num or tok.is_punct or tok.is_space or tok.has_vector):
                log.debug('{}: _unknown_words_match: token {}'.format(type(self).__name__, repr(tok.text)))
                return True
        return False


class HeuristicSpanTagger(HeuristicTagger):
    def tag(self, span: Span):
        res1 = self._ner_match(span)
        # res2 = self._unknown_words_match(span)
        res3 = self._similar_words_match(span)
        # res = res1 or res2 or res3
        res = res3 or res1

        #res = self._ner_match(span) or \
              # self._unknown_words_match(span) or \
              # self._similar_words_match(span)

        _tag = None if res else self.tags.default_tag
        log.info('{}: tag: tag {}; text {}'.format(type(self).__name__, _tag, repr(span.text)))
        return _tag


class HeuristicEventTagger(HeuristicTagger):
    def tag(self, event: EventOffsets):
        # we want to have all extracted tokens to be related to one doc
        # only that way we can investigate connections (e.g. dependency tree links) between them
        doc = self.nlp(event.sentence)
        ent1_tokens = self._extract_tokens(event.entity1_offsets, doc)
        action_tokens = self._extract_tokens(event.action_offsets, doc)
        ent2_tokens = self._extract_tokens(event.entity2_offsets, doc)

        # now is the time to use heuristics; we apply:
        # - named entity type matcher to entity1 and entity2 (there unlikely something interesting in action)
        # - keyword matcher for all tokens in event
        # - matcher of the unknown words: words that has no word embedding
        res = self._ner_match(ent1_tokens + ent2_tokens) or \
              self._unknown_words_match(ent1_tokens + ent2_tokens) or \
              self._similar_words_match(action_tokens + ent1_tokens + ent2_tokens)

        # that tagger is careful: it gives up when finds something
        # only when it finds nothing, it assigns default tag (e.g. '0' - category of negative examples)
        # return None if res else self.tags.default_tag
        _tag = None if res else self.tags.default_tag
        log.info('{}: tag: tag={}; event={}'.format(type(self).__name__, _tag, repr(event)))
        return _tag

    def _extract_tokens(self, offsets, doc):
        """Extract tokens from doc which fall in specified offsets.
        In other words, map offsets to actual tokens."""
        return [tok for tok in doc if any((offset[0] <= tok.idx < offset[1]) for offset in offsets)]
