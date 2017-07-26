

class Tags:
    """Mapping between tags in user-format and format used in algorithms."""
    def encode(self, raw_tag):
        raise NotImplementedError

    def decode(self, cat):
        raise NotImplementedError

    def __contains__(self, raw_tag):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class CategoricalTags(Tags):
    def __init__(self, raw_tags, add_default_tag=False):
        """
        :param raw_tags: tags for mapping to numerical values
        :param add_default_tag: bool. If True, add additional 'fallback' tag (which takes zero class).
        """
        i = int(add_default_tag)
        nbtags = len(raw_tags) + i
        self.tag_map = dict(zip(raw_tags, range(i, nbtags)))
        self.default_tag = 0 if add_default_tag else None
        # Case when there're only 2 values to distinguish. Only one value is necessary (i.e. 0-1 vs. [0 1]-[1 0]).
        self.nbtags = 1 if nbtags == 2 else nbtags

    def encode(self, raw_tag):
        index = self.tag_map.get(raw_tag, self.default_tag)
        if index is None:
            raise KeyError('Invalid (unknown) tag provided and default tag is not set: {}'.format(raw_tag))
        if self.nbtags > 1:
            cat = [0] * self.nbtags
            cat[index] = 1
            return cat
        return index

    def decode(self, cat):
        i = cat.index(1) if isinstance(cat, list) else cat
        try:  # make exception expecting explicit
            return self.tag_map[i]
        except TypeError:
            raise TypeError('Category has inappropriate type. Expected list or int. Got {}'.format(type(cat)))

    def __contains__(self, item):
        return item in self.tag_map

    def __len__(self):
        return self.nbtags

    __getitem__ = encode


def categorical_tags_tests():
    tags1 = CategoricalTags(('first',), True)
    assert('zero' not in tags1)
    assert(tags1.default_tag is not None)
    tags2 = CategoricalTags(('zero','first',), False)
    assert('zero' in tags2)
    assert(tags2.default_tag is None)

    assert(tags1.encode('zero') == tags2.encode('zero'))
    assert(len(tags1) == len(tags2) == 1)
    assert(tags1.encode('zero') == tags1['zero'])


if __name__ == "__main__":
    categorical_tags_tests()
