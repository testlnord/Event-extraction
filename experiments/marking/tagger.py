import logging as log
from spacy.tokens import Span


class Tags:
    def encode(self, raw_tag) -> list:
        raise NotImplementedError

    def decode(self, cat):
        raise NotImplementedError

    def is_correct(self, raw_or_cat) -> bool:
        raise NotImplementedError


class Tagger:
    def __init__(self, tags: Tags):
        self.tags = tags

    def __call__(self, texts):
        for text in texts:
            yield text, self.tag(text)

    def tag(self, text: Span):
        raise NotImplementedError


class DummyTagger(Tagger):
    def tag(self, text):
        return str(0)


class UserTagger(Tagger):
    def __init__(self, tags: Tags, escape_input=''):
        self.end = escape_input
        super(UserTagger, self).__init__(tags)

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

    def _to_categorical(self, raw_tag):
        cat = [0] * self.nbtags
        cat[self.raw_tags.index(str(raw_tag))] = 1
        return cat

    def _to_raw(self, cat) -> str:
        i = cat.index(1)
        return self.raw_tags[i]
