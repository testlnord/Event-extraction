

class Tags:
    """Mapping between tags in user-format and format used in algorithms."""
    def encode(self, raw_tag):
        raise NotImplementedError

    def decode(self, cat):
        raise NotImplementedError

    def is_correct(self, raw_tag):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @property
    def default_tag(self):
        return None


class CategoricalTags(Tags):
    def __init__(self, raw_tags):
        """
        :param raw_tags: must allow conversion to string without loss of information
        """
        self.raw_tags = tuple(map(str, raw_tags))
        nbtags = len(raw_tags)
        # Case when there're only 2 values to distuinguish
        self.nbtags = 1 if nbtags == 2 else nbtags

    def encode(self, raw_tag):
        index = self.raw_tags.index(str(raw_tag))
        if self.nbtags > 1:
            cat = [0] * self.nbtags
            cat[index] = 1
            return cat
        return index

    def decode(self, cat):
        i = None
        if isinstance(cat, list):
            i = cat.index(1)
        elif isinstance(cat, int):
            i = cat
        else:
            raise TypeError('Category has inappropriate type. Expected list or int. Got {}'.format(type(cat)))
        return self.raw_tags[i]

    def is_correct(self, raw_tag):
        return raw_tag in self.raw_tags

    @property
    def default_tag(self):
        return '0'

    def __len__(self):
        return self.nbtags
