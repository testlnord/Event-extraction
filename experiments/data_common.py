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


def split(slicible, batch_size=1, splits=(0.8, 0.2)):
    """Split something on len(splits) parts with sizes proportional values in splits.
    Sizes of the parts will be multiples of batch_size."""
    edges = split(len(slicible), batch_size, splits)
    subsets = [slicible[a:b] for a, b in edges]
    return subsets


def batch(iterable, batch_size=1, element_wise=True):
    """Yields tuples of batchs (element_wise == True) or batch of tuples (element_wise == False)"""
    cur_batch = []
    for elems in iterable:
        cur_batch.append(elems)
        if len(cur_batch) == batch_size:
            # dimension checking
            # print('batch_shape_before_reshape (expect ({}, {})):'.format(batch_size, len(elems)), np.shape(cur_batch))
            l = len(elems)
            yield tuple([np.array([cur_batch[j][i] for j in range(batch_size)]) for i in range(l)]) if element_wise else cur_batch
            cur_batch = []

