import logging as log
import numpy as np


def split_range(length, batch_size, splits):
    """Splits range of length=length to subranges of size even to batch_size
    and returns indexes of that subranges in the original range"""
    l = length
    # throw points that just don't fit in the last batch
    parts = l // batch_size
    l = parts * batch_size
    sub_lens = map(lambda x: int(round(x * parts, 0)) * batch_size, splits)

    edges = []
    b = 0
    for split in sub_lens:
        a = b
        b += split
        edges.append((a, b))
    # last edge is the last index
    edges[-1] = edges[-1][0], l

    log.debug('Splitting length={} with batch_size={}.'.format(length, batch_size) +
              "Probably discarding last {} data points that don't fit in the last batch:".format(length-l))
    return edges


def split(slicible, batch_size, splits):
    """Split something on len(splits) parts with sizes proportional values in splits.
    Sizes of the parts will be multiples of batch_size."""
    edges = split(len(slicible), batch_size, splits)
    subsets = [slicible[a:b] for a, b in edges]
    return subsets


def visualise(model, filename='model.png', exit_after=False):
    from keras.utils.visualize_util import plot
    plot(model, show_shapes=True, to_file=filename)
    log.info('Saved visualisation of the model')
    if exit_after:
        exit()


class BatchMaker:
    def __init__(self, batch_size=1, element_wise_batches=True):
        self.batch_size = batch_size
        self.element_wise = element_wise_batches

    def __call__(self, iterable):
        """Yields tuples of batchs (element_wise == True) or batch of tuples (element_wise == False)"""
        cur_batch = []
        for elems in iterable:
            cur_batch.append(tuple(map(np.array, elems)))
            if len(cur_batch) == self.batch_size:
                l = len(elems)
                # transposing if element_wise
                # in essence: res = zip(*cur_batch); and type conversions
                res = list(map(np.array, zip(*cur_batch))) if self.element_wise else cur_batch
                yield res
                cur_batch.clear()


class Padding:
    def __init__(self, pad_to_length, pad_value=0, pad_tags=False, cut_too_long=True):
        self.pad_to_length = pad_to_length
        self.pad_value = pad_value
        self.pad_tags = pad_tags
        self.cut_too_long = cut_too_long

    def __call__(self, args_generator):
        for args in args_generator:
            yield self.pad(*args[:2])

    def pad(self, text, tag):
        _text = np.array(text)
        _tag = np.array(tag)

        ones = len(text)
        zeros = max(0, self.pad_to_length - ones)
        pad_text = ((0, zeros), (0, 0))
        cut = min(ones, self.pad_to_length) if self.cut_too_long else ones

        # mask, telling us about inserted redundant values
        sample_weights = np.array([1] * cut + [0] * zeros)
        _text = np.pad(_text[:cut], pad_width=pad_text, mode='constant', constant_values=self.pad_value)
        if self.pad_tags:
            _tag = np.pad(_tag[:cut], pad_width=pad_text, mode='constant', constant_values=self.pad_value)

        return _text, _tag, sample_weights
