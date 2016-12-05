
class Encoder:
    def __call__(self, data_iterable):
        for data in data_iterable:
            if isinstance(data, tuple) and len(data) == 2:
                yield self.encode_data(data[0]), self.encode_tag(data[1])
            else:
                yield self.encode_data(data)

    def encode_data(self, data):
        raise NotImplementedError

    def encode_tag(self, tag):
        raise NotImplementedError

