"""
Reduction of slice objects
"""
# Author: Pearu Peterson
# Created: June 2020


def reduced(s, size):
    """Return reduced slice with respect to given size.
    Notes
    -----
    If `x = list(range(size))` then `x[s] == x[reduced(s, size)]` for
    any slice instance `s`.
    """
    start, stop, step = s.start, s.stop, s.step
    if step is None:
        step = 1
    if step > 0:
        if start is None or start < -size:
            start = 0
        elif start < 0:
            start += size
        elif start > size:
            start = size
        if stop is None or stop > size:
            stop = size
        elif stop < -size:
            stop = 0
        elif stop < 0:
            stop += size
        if step > size:
            step = size
    else:
        if start is None or start >= size:
            start = -1
        elif start < -size:
            start = -size - 1
        elif start >= 0:
            start -= size
        if stop is None or stop < -size:
            stop = -size - 1
        elif stop >= size:
            stop = -1
        elif stop >= 0:
            stop -= size
        if step < -size:
            step = -size
    if (stop - start) * step <= 0:
        start = stop = 0
        step = 1
    return slice(start, stop, step)


def nitems(s, size, reduce=True):
    """Return the number of items for given size.

    Notes
    -----
    1. If `x = list(range(size))` and `s = reduced(s, size)` then
      `len(x[s]) == nitems(s, size)` for any slice instance `s`.

    2. If `s = reduced(s, size)` then `x[s] = [(s.start + i * s.step)
      % size for i in range(nitems(s, size))]`.
    """
    if reduce:
        s = reduced(s, size)
    if s.stop == s.start:
        return 0
    sgn = s.step // abs(s.step)
    return max(0, ((s.stop - s.start) + s.step - sgn) // s.step)


def apply_slices(slices, shape, strides):
    """Apply slices to array shape and strides. Return new shape,
    strides, and offset."""

    new_shape = []
    new_strides = []
    offset = 0
    for size, stride, s in zip(shape, strides, slices):
        if isinstance(s, int):
            offset += stride * (s % size)
        elif isinstance(s, slice):
            s = reduced(s, size)
            offset += stride * (s.start % size)
            new_strides.append(stride * s.step)
            new_shape.append(nitems(s, size, reduce=False))
        elif s is None:  # None represents slice(None, None, None)
            new_strides.append(stride)
            new_shape.append(size)
        else:
            raise ValueError(f'slices items must be int or slice, got {type(s)}')
    return tuple(new_shape), tuple(new_strides), offset


def shape_strided(index, shape):
    new_shape = []
    for ind, size in zip(index, shape):
        if isinstance(ind, slice):
            new_shape.append(nitems(ind, size, reduce=False))
        elif ind is None:
            new_shape.append(size)
    return tuple(new_shape)


def index_strided(index, shape, strides, offset=0):
    """Index strided array, return new shape, strides and offset.

    The index must be normalized.
    """
    new_shape, new_strides = [], []
    for i, (ind, size, stride) in enumerate(zip(index, shape, strides)):
        if isinstance(ind, int):
            offset += stride * ind
        elif isinstance(ind, slice):
            offset += stride * (ind.start % size)
            new_shape.append(nitems(ind, size, reduce=False))
            new_strides.append(stride * ind.step)
        elif ind is None:
            new_shape.append(size)
            new_strides.append(stride)
        else:
            raise ValueError(f'index items must be int or slice or None, got {type(s)} at index {i}')
    return tuple(new_shape), tuple(new_strides), offset


def normalize_index(index, shape):
    """Reduce slice indices and normalize negative integer indices.
    """
    if isinstance(index, int):
        return (index % shape[0],) + (None,) * (len(shape)-1)
    elif isinstance(index, slice):
        return (reduced(index, shape[0]),) + (None,) * (len(shape)-1)
    elif index is None:
        return (None,) * len(shape)
    new_index = [None] * len(shape)
    for i in range(len(index)):
        ind = index[i]
        if isinstance(ind, int):
            new_index[i] = ind % shape[i]
        elif isinstance(ind, slice):
            new_index[i] = reduced(ind, shape[i])
        elif ind is None:
            pass
        else:
            raise ValueError(f'index items must be int or slice or None, got {type(ind)} at index {i}')
    return tuple(new_index)
