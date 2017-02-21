

class Tags:
    def encode(self, raw_tag):
        raise NotImplementedError

    def decode(self, cat):
        raise NotImplementedError

    def is_correct(self, raw_tag):
        raise NotImplementedError

    @property
    def default_tag(self):
        """Allows to return default tag, for example, class representing all negative examples.
        None otherwise (by default)."""
        return None


class CategoricalTags(Tags):
    """Provided tags must allow conversion to string without loss of information"""
    def __init__(self, raw_tags):
        self.raw_tags = tuple(map(str, raw_tags))
        self.nbtags = len(raw_tags)

    def encode(self, raw_tag):
        return self._to_categorical(raw_tag)

    def decode(self, cat):
        return self._to_raw(cat)

    def is_correct(self, raw_tag):
        return raw_tag in self.raw_tags

    @property
    def default_tag(self):
        """Returns default raw tag"""
        return '0'

    def _to_categorical(self, raw_tag):
        cat = [0] * self.nbtags
        cat[self.raw_tags.index(str(raw_tag))] = 1
        return cat

    def _to_raw(self, cat):
        i = None
        if isinstance(cat, list):
            i = cat.index(1)
        elif isinstance(cat, int):
            i = cat
        else:
            raise TypeError('Category has inappropriate type. Expected list or int. Got {}'.format(type(cat)))

        return self.raw_tags[i]


