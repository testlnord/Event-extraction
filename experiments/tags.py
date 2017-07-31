

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
    def __init__(self, raw_tags, default_tag=None):
        """
        :param raw_tags: tags for mapping to numerical values
        :param add_default_tag: bool. If True, add additional 'fallback' tag (which takes zero class).
        """
        i = int(default_tag is not None)
        nbtags = len(raw_tags) + i
        self.tag_map = dict(zip(raw_tags, range(i, nbtags)))
        self.inv_tag_map = dict(zip(range(i, nbtags), raw_tags))
        self._default_tag_index = None
        if i != 0:
            self._default_tag_index = 0
            self.inv_tag_map[0] = default_tag
        self.default_tag = default_tag  # it is a raw_tag
        # Case when there're only 2 values to distinguish. Only one value is necessary (i.e. 0-1 vs. [0 1]-[1 0]).
        self.nbtags = 1 if nbtags == 2 else nbtags

    def encode(self, raw_tag):
        index = self.tag_map.get(raw_tag, self._default_tag_index)
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
            return self.inv_tag_map[i]
        except TypeError:
            raise TypeError('Category has inappropriate type. Expected list or int. Got {}'.format(type(cat)))

    def __contains__(self, item):
        return item in self.tag_map

    def __len__(self):
        return self.nbtags

    __getitem__ = encode


def categorical_tags_tests():
    tags1 = CategoricalTags(('one',), 'None')
    assert('zero' not in tags1)
    assert(tags1._default_tag_index is not None)
    tags2 = CategoricalTags(('zero','one',))
    assert('zero' in tags2)
    assert(tags2._default_tag_index is None)

    assert(tags1.encode('zero') == tags2.encode('zero'))
    assert(len(tags1) == len(tags2) == 1)
    assert(tags1.encode('zero') == tags1['zero'])

    assert(tags2.encode('zero') == 0)
    assert(tags2.decode(0) == 'zero')
    assert(tags1.encode('zero') == 0)
    assert(tags1.decode(0) == 'None')


if __name__ == "__main__":
    categorical_tags_tests()
