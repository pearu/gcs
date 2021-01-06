"""Implementaions of various storage objects for multidimensional arrays:

Buffer - a 1-dimensional contiguous array
Strided - a multidimensional array with strides
COO - a multidimensional array using Coordinate format
FlatCOO - a multidimensional array using Coordinate format with flattened indices
CRS - a 2-dimensional array using Compressed Row Storage format
GCS - a mulitdimensional array using Generalized Compressed Storage based on dimensionality reduction 
"""
# Author: Pearu Peterson
# Created: December 2020

import itertools
import math
import sys
from . import slice_tools


DEFAULT = object()


def compress_indices(indices, size):
    """Return compressed indices.

    The function is used for compressing COO row indices to CSR
    compressed representation. The input indices must be sorted.
    """
    nse = len(indices)
    compressed = [0] * (size + 1)

    k = 1
    last_index = 0
    for i in range(nse):
        index = indices[i]
        for n in range(last_index, index):
            compressed[k] = i
            k += 1
        last_index = index

    for n in range(k, size + 1):
        compressed[n] = nse

    return compressed


def uncompress_indices(compressed):
    """Reverse of compress_indices.
    """
    indices = []
    for i in range(len(compressed) - 1):
        indices.extend([i] * (compressed[i+1] - compressed[i]))
    return indices


def product(seq):
    """Product of sequence items.
    """
    if sys.version_info[:2] >= (3, 8):
        return math.prod(seq)
    return seq[0] * product(seq[1:]) if seq else 1


def make_strides(shape):
    """Return C-contiguous strides from shape.
    """
    return tuple(product(shape[i + 1:]) for i in range(len(shape)))


def get_shape(seq):
    """Estimate the shape of a nested sequence.
    """
    if isinstance(seq, Storage):
        return seq.shape
    if isinstance(seq, (tuple, list)):
        if seq:
            return (len(seq), ) + get_shape(seq[0])
        return (0,)
    return ()


def get_value(seq, index):
    """Get the value of a nested sequence using index.
    """
    if index == ():
        return seq
    if isinstance(seq, Storage):
        return seq[index]
    assert isinstance(seq, (tuple, list)), type(seq)
    value = seq
    for i in index:
        value = value[i]
    return value


def is_element_index(index, shape):
    """Return True if index is a concrete index of nested sequence with
    given shape. Return False if index is an incomplete index or slice
    such that indexing operation would result another nested sequence
    rather than an element value.
    """
    if isinstance(index, slice) or index is None:
        return False
    if isinstance(index, int):
        return len(shape) == 1
    if len(index) == len(shape):
        for ind in index:
            if not isinstance(ind, int):
                return False
        return True
    return False

def get_unspecified(seq, unspecified=DEFAULT):
    """Return a value that represents unspecified element of a nested
    sequence.
    """
    if isinstance(seq, Storage):
        return seq.unspecified
    return None


def get_rposition(index, strides, shape, axis_map):
    d = len(shape)
    if d == 0:
        return 0
    if d == 1:
        return index[axis_map[0]] * strides[0]
    if d == 2:
        return index[axis_map[0]] * strides[0] + index[axis_map[1]] * strides[1]
    if d == 3:
        return index[axis_map[0]] * strides[0] + index[axis_map[1]] * strides[1] + index[axis_map[2]] * strides[2]
    if d == 4:
        return index[axis_map[0]] * strides[0] + index[axis_map[1]] * strides[1] + index[axis_map[2]] * strides[2] + index[axis_map[3]] * strides[3]
    if d == 5:
        return index[axis_map[0]] * strides[0] + index[axis_map[1]] * strides[1] + index[axis_map[2]] * strides[2] + index[axis_map[3]] * strides[3] + index[axis_map[4]] * strides[4]
    return sum(index[axis_map[i]] * strides[i] for i in range(len(shape)))


def reduce_index(rshapes, rdimensions, rstrides, roffset, index):
    """Return the index of dimensionality reduced array that corresponds
    to the given index of the original array.
    """
    return tuple(roffset[k] + get_rposition(index, rstrides[k], rshapes[k], rdimensions[k])
                 for k in range(len(rshapes)))


def promote_index(rshapes, rdimensions, rstrides, roffset, rindex):
    """Return the index of the original array that corresponds to the
    given index of the dimensionality reduced array.
    """
    dimensionality = sum(map(len, rdimensions))
    index = [None] * dimensionality
    if isinstance(rindex, int):
        assert len(rdimensions) == 1
        pos_0 = rindex - roffset[0]
        rstrides_0 = rstrides[0]
        rdimensions_0 = rdimensions[0]
        rshapes_0 = rshapes[0]
        for i in range(len(rdimensions[0])):
            index[rdimensions_0[i]] = (pos_0 // rstrides_0[i]) % rshapes_0[i]
    else:
        for k in range(len(rshapes)):
            pos_k = rindex[k] - roffset[k]
            rstrides_k = rstrides[k]
            rdimensions_k = rdimensions[k]
            rshapes_k = rshapes[k]
            for i in range(len(rdimensions[k])):
                index[rdimensions_k[i]] = (pos_k // rstrides_k[i]) % rshapes_k[i]
    return tuple(index)


def get_index(pos, shape, strides):
    """Return the index of the original array that corresponds to the
    index (pos) of an one-dimensional dimensionality reduced array.
    """
    d = len(shape)
    if d == 0:
        return ()
    elif d == 1:
        return ((pos // strides[0]) % shape[0],)
    elif d == 2:
        return ((pos // strides[0]) % shape[0], (pos // strides[1]) % shape[1])
    elif d == 3:
        return ((pos // strides[0]) % shape[0], (pos // strides[1]) % shape[1], (pos // strides[2]) % shape[2])
    elif d == 4:
        return ((pos // strides[0]) % shape[0], (pos // strides[1]) % shape[1], (pos // strides[2]) % shape[2], (pos // strides[3]) % shape[3])
    elif d == 5:
        return ((pos // strides[0]) % shape[0], (pos // strides[1]) % shape[1], (pos // strides[2]) % shape[2], (pos // strides[3]) % shape[3], (pos // strides[4]) % shape[4])
    return tuple((pos // strides[i]) % shape[i] for i in range(d))


def get_position(index, shape, strides):
    """Return the index (pos) of one-dimensional dimensionality reduced
    array that corresponds to the given index of the original array.
    """
    d = len(shape)
    if d == 0:
        return 0
    elif d == 1:
        return index[0] * strides[0]
    elif d == 2:
        return index[0] * strides[0] + index[1] * strides[1]
    elif d == 3:
        return index[0] * strides[0] + index[1] * strides[1] + index[2] * strides[2]
    elif d == 4:
        return index[0] * strides[0] + index[1] * strides[1] + index[2] * strides[2] + index[3] * strides[3]
    elif d == 5:
        return index[0] * strides[0] + index[1] * strides[1] + index[2] * strides[2] + index[3] * strides[3] + index[4] * strides[4]
    return sum(index[i] * strides[i] for i in range(d))


def get_slice_index(index, shape, strides, slice_shape, slice_strides, slice_offset):
    """Return the index of sliced array that corresponds to the given
    index of the original array. Return None if the element of the
    original array is not a part of the sliced array.
    """
    strided_pos = get_position(index, shape, strides) - slice_offset
    slice_index = get_index(strided_pos, slice_shape, slice_strides)
    spos = get_position(slice_index, slice_shape, slice_strides)
    return slice_index if spos == strided_pos else None


def get_index_slice(slice_index, shape, strides, slice_shape, slice_strides, slice_offset):
    """Return the index of original array that corresponds to the given
    index of sliced array.
    """
    spos = get_position(slice_index, slice_shape, slice_strides) + slice_offset
    return get_index(spos, shape, strides)


def get_coo_position(index, shape, indices):
    """Return the position of index in COO indices. If index is not in
    indices, return None.
    """
    # binary search
    l = 0
    r = indices.shape[1] - 1
    dims = len(shape)
    while l <= r:
        pos = (l + r) // 2
        for j in range(dims):
            d = indices[j, pos] - index[j]
            if d == 0:
                continue
            if d < 0:
                l = pos + 1
            elif d > 0:
                r = pos - 1
            break
        else:
            return pos
    # index is not in indices
    return


def get_flatcoo_position(findex, shape, findices):
    """Return the position of index in flattend COO indices. If index is
    not in indices, return None.
    """
    # binary search
    l = 0
    r = len(findices) - 1
    dims = len(shape)
    while l <= r:
        pos = (l + r) // 2
        d = findices[pos] - findex
        if d == 0:
            return pos
        if d < 0:
            l = pos + 1
        elif d > 0:
            r = pos - 1
    # findex is not in findices
    return


def get_crs_position(index, shape, crow_indices, col_indices):
    """Return the position of index in CRS indices representation. If
    index is not in indices, return None.
    """
    # binary search
    row, col = index
    l = crow_indices[row]
    r = crow_indices[row+1] - 1
    while l <= r:
        i = (l + r) // 2
        d = col_indices[i] - col
        if d == 0:
            return i
        if d < 0:
            l = i + 1
        elif d > 0:
            r = i - 1
    # index is not in indices
    return


class Storage:
    """Base class for storage implementations.

    As a minimum, subclasses must define the following instance
    attributes:

      shape    - shape of an array
      data     - Storage instance

    and the following methods:

      fromobject - class method
      __init__
      _get_object_data
      indices_and_data_position
    """

    def __len__(self):
        return self.shape[0]

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self is other or self._get_object_data() == other._get_object_data()
        return NotImplemented

    def is_full(self):
        return self.size == product(self.shape)

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return (f'{type(self).__name__}{self._get_object_data()}')

    def _get_object_data(self):
        """Return a tuple of constructor arguments. The last item must be Storage instance.
        """
        raise NotImplementedError(f'{type(self).__name__}._get_object_data')

    def copy(self, unspecified=DEFAULT):
        """Copy of the storage.
        """
        obj_data = self._get_object_data()
        copy_data = obj_data[:-1] + (obj_data[-1].copy(unspecified=unspecified),)
        return type(self)(*copy_data)

    @property
    def unspecified(self):
        return self.data.unspecified

    @property
    def strides(self):
        s = getattr(self, '_strides', DEFAULT)
        if s is DEFAULT:
            s = self._strides = make_strides(self.shape)
        return s

    def equal(self, other):
        """Element-wise equality.
        """
        d1, d2 = {}, {}
        for index, value in self.elements():
            d1[index] = value
        if not isinstance(other, Storage):
            if isinstance(self, Buffer):
                other = type(self).fromobject(other, unspecified=self.unspecified)
            else:
                other = Strided.fromobject(other, unspecified=self.unspecified)
        for index, value in other.elements():
            d2[index] = value
        if d1 != d2:
            print(f'{d1=}')
            print(f'{d2=}')
        return d1 == d2

    def __setitem__(self, index, value):
        index = slice_tools.normalize_index(index, self.shape)
        shape, strides, offset = slice_tools.index_strided(index, self.shape, self.strides)
        for value_index, data_pos in self.slice_indices_and_data_position(shape, strides, offset):
            self.data[data_pos] = get_value(value, value_index)

    def get_data_position(self, index):
        raise NotImplementedError(f'{type(self).__name__}.get_data_position')

    def get_slice(self, shape, strides, offset):
        raise NotImplementedError(f'{type(self).__name__}.get_slice')

    def __getitem__(self, index):
        index = slice_tools.normalize_index(index, self.shape)
        if is_element_index(index, self.shape):
            data_pos = self.get_data_position(index)
            if data_pos is None:
                return self.unspecified
            return self.data[data_pos]
        shape, strides, offset = slice_tools.index_strided(index, self.shape, self.strides)
        return self.get_slice(shape, strides, offset)

    def indices_and_data_position(self):
        raise NotImplementedError(f'{type(self).__name__}.indices_and_data_position')

    def slice_indices_and_data_position(self, shape, strides, offset, sort=False):
        """Iterator of sliced array elements as slice index and data position pairs.

        The inputs represents the shape, strides, and offset of the sliced array.
        """
        if sort:
            # Optimal method when the data storage is strided array or
            # when the number of sliced array elements is considerably
            # smaller than of the original array.
            # Required for slicing CSR/COO formatted arrays that must have indices sorted. 
            for slice_index in itertools.product(*map(range, shape)):
                index = get_index_slice(slice_index, self.shape, self.strides, shape, strides, offset)
                data_pos = self.get_data_position(index)  # this is expensive for CSR and COO arrays!
                yield slice_index, data_pos
        else:
            # Optimal method when the size of sliced array is in the
            # same order of magnitude as the size of the original
            # array
            for index, data_pos in self.indices_and_data_position():
                slice_index = get_slice_index(index, self.shape, self.strides, shape, strides, offset)
                if slice_index is not None:
                    yield slice_index, data_pos

    def elements(self):
        """Iterator of array elements represented as index and value pairs.
        """
        for index, data_pos in self.indices_and_data_position():
            yield index, self.data[data_pos]

    def __str__(self):
        if not isinstance(self, Strided):
            return repr(self)
        if len(self.shape) == 1:
            line = ', '.join([str(self[i]) for i in range(self.shape[0])])
            if len(line) > 100:
                line = line[:40] + "....." + line[-40:]
            return f'[{line}]'

        if len(self.shape) == 2:
            rows = '\n  '.join([str(self[i]) for i in range(self.shape[0])])
            return f'[ {rows}\n]'
        return repr(self)

    # storage conversion methods

    def as_Buffer(self, unspecified=DEFAULT):
        return Buffer.fromobject(self, unspecified=unspecified)

    def as_Strided(self, unspecified=DEFAULT):
        return Strided.fromobject(self, unspecified=unspecified)

    def as_COO(self, unspecified=DEFAULT):
        return COO.fromobject(self, unspecified=unspecified)

    def as_FlatCOO(self, unspecified=DEFAULT):
        return FlatCOO.fromobject(self, unspecified=unspecified)

    def as_CRS(self, unspecified=DEFAULT):
        return CRS.fromobject(self, unspecified=unspecified)

    def as_GCS(self, dimensions=None, partitioning=None, storage_class=None, unspecified=DEFAULT):
        return GCS.fromobject(
            self, shape=self.shape, dimensions=dimensions,
            partitioning=partitioning, storage_class=storage_class,
            unspecified=unspecified)


class Buffer(Storage):
    """Represents a contiguous one-dimensional array that is shared by
    storage objects from slicing operations.
    """

    @classmethod
    def fromobject(cls, seq, shape=None, get_seq_index=None, unspecified=DEFAULT):
        if isinstance(seq, Buffer) and unspecified is DEFAULT and shape is None and get_seq_index is None:
            return seq
        seq_unspecified = get_unspecified(seq)
        if unspecified is DEFAULT:
            unspecified = seq_unspecified
        if get_seq_index is None:
            get_seq_index = lambda index: index
        if shape is None:
            shape = get_shape(seq)
        values = []
        for index in itertools.product(*map(range, shape)):
            value = get_value(seq, get_seq_index(index))
            if value == seq_unspecified:
                value = unspecified
            values.append(value)
        return cls(len(values), 0, values, unspecified=unspecified)

    def __init__(self, size:int, offset:int, data, unspecified=None):
        """Parameters
        ----------
        size : int
          Buffer size
        offset: int
          Indexing offset with respect to values
        data : sequence
          A sequence of values. It can be any object that supports
          indexing operations for indexes within `range(offset,
          offset+size)` and that implements copy method.
        unspecified: object
          Place-holder or value of an unspecified elements.
        """
        assert isinstance(size, int), size
        assert isinstance(offset, int), offset
        assert offset + size <= len(data), (offset, size, len(data))
        assert not isinstance(data, Storage)
        self.shape = (size,)
        self.size = size
        self.offset = offset
        self.data = data
        self._unspecified = unspecified

    def _get_object_data(self):
        return self.size, self.offset, self.unspecified
        
    @property
    def unspecified(self):
        return self._unspecified

    def copy(self, unspecified=DEFAULT):
        """Return a copy of the buffer.
        """
        if unspecified is DEFAULT:
            unspecified = self.unspecified
        if unspecified == self.unspecified:
            data = self.data[self.offset:self.offset+self.size].copy()
        else:
            data = []
            for pos in range(self.offset, self.offset + self.size):
                value = self.data[pos]
                if value == self.unspecified:
                    value = unspecified
                data.append(value)
        return Buffer(len(data), 0, data, unspecified=unspecified)

    def __repr__(self):
        data = str(self.data[self.offset:self.offset+self.size])
        n = self.offset
        p = "*" * (n//10) + "." * (n % 10)
        n = len(self.data)-(self.offset+self.size)
        s = "." * (n%10) + "*" * (n//10)
        if len(data)>100:
            data = data[:40] + "....." + data[-40:]
        return (f'{type(self).__name__}({self.size}, {self.offset}, {p}{data}{s}, unspecified={self.unspecified})')

    def __eq__(self, other):
        # object equality modulo data and unspecified arbitrariness
        if isinstance(other, type(self)):
            return (self.size == other.size
                    and (self.data[self.offset:self.offset+self.size]
                         == other.data[other.offset:other.offset+other.size])
                    and self.unspecified==other.unspecified)
        return NotImplemented

    def _get_complete_index(self, index):
        if isinstance(index, tuple):
            assert len(index) == 1
            return index[0]
        return index

    def __getitem__(self, index):
        index = self._get_complete_index(index)
        if index is None:
            return self
        # Warning: negative indices are interpreted with respect to
        # offset! It means that the result of slicing can be larger
        # than the original buffer.
        if isinstance(index, slice):
            start = self.offset + index.start if index.start is not None else self.offset
            stop = self.offset + index.stop if index.stop is not None else self.offset + self.size
            if start < 0 or start >= len(self.data):
                raise IndexError(f'{type(self).__name__} start position must be in range({len(self.data)}), got {start}')
            if stop <= 0 or stop > len(self.data):
                raise IndexError(f'{type(self).__name__} stop position must be in range(1, {len(self.data)+1}), got {stop}')
            s = slice(start, stop, index.step)
            s = slice_tools.reduced(s, len(self.data))
            if s.step != 1:
                raise IndexError(f'{type(self).__name__} supports slicing with step 1 only, got {s.step}')
            size = s.stop - s.start
            return Buffer(size, s.start, self.data, unspecified=self.unspecified)

        pos = self.offset + index
        return self.data[pos]

    def __setitem__(self, index, value):
        index = self._get_complete_index(index)
        assert isinstance(index, int), index  ## TODO: slice support
        pos = self.offset + index
        self.data[pos] = value

    def __iter__(self):
        for pos in range(self.offset, self.offset + self.size):
            yield self.data[pos]

    def indices_and_data_position(self):
        for data_pos in range(self.offset, self.offset + self.size):
            index = data_pos - self.offset
            yield index, data_pos


class Strided(Storage):
    """Represents a multidimensional array with strides.
    """
    
    @classmethod
    def fromobject(cls, seq, shape=None, get_seq_index=None, unspecified=DEFAULT):
        if isinstance(seq, Strided) and shape is None and get_seq_index is None and unspecified is DEFAULT:
            return seq
        if shape is None:
            shape = get_shape(seq)
        data = Buffer.fromobject(seq, shape=shape, get_seq_index=get_seq_index, unspecified=unspecified)
        return cls(shape, make_strides(shape), data)

    def __init__(self, shape: tuple, strides: tuple, data: Buffer):
        assert isinstance(data, Buffer)
        assert len(shape) == len(strides)
        self.shape = shape
        self._strides = strides
        self.data = data
        self.size = product(shape)

    def _get_object_data(self):
        return self.shape, self.strides, self.data
        
    def copy(self, unspecified=DEFAULT):
        """Return a row-wise contiguous copy.
        """
        if unspecified is DEFAULT:
            unspecified = self.unspecified
        data = Buffer(self.size, 0, [unspecified] * self.size, unspecified=unspecified)
        result = Strided(self.shape, make_strides(self.shape), data)
        for index, value in self.elements():
            if value == self.unspecified:
                value = unspecified
            result[index] = value
        return result

    def indices_and_data_position(self):
        for index in itertools.product(*map(range, self.shape)):
            data_pos = get_position(index, self.shape, self.strides)
            yield index, data_pos

    def slice_indices_and_data_position(self, shape, strides, offset, sort=False):
        # Optimal method for strided arrays
        for slice_index in itertools.product(*map(range, shape)):
            data_pos = offset + get_position(slice_index, shape, strides)
            yield slice_index, data_pos

    def get_data_position(self, index):
        return get_position(index, self.shape, self.strides)

    def get_slice(self, shape, strides, offset):
        return Strided(shape, strides, self.data[offset:])


class COO(Storage):
    """Represents a multidimensional array using COO storage format.
    """
    @classmethod
    def fromobject(cls, seq, shape=None, get_seq_index=None, unspecified=DEFAULT):
        if isinstance(seq, COO) and unspecified is DEFAULT and shape is None and get_seq_index is None:
            return seq
        if isinstance(seq, FlatCOO) and unspecified is DEFAULT and shape is None and get_seq_index is None:
            shape = seq.shape
            indices = [[] for d in shape]
            for findex in seq.findices:
                index = get_index(findex, shape, seq.strides)
                for i in range(len(shape)):
                    indices[i].append(index[i])
            indices = Strided.fromobject(indices)
            return cls(shape, indices, seq.data.copy())
        # TODO: seq is CSR
        seq_unspecified = get_unspecified(seq)
        if unspecified is DEFAULT:
            unspecified = seq_unspecified
        if shape is None:
            shape = get_shape(seq)
        if get_seq_index is None:
            get_seq_index = lambda index: index
        indices = [[] for d in shape]
        values = []
        for index in itertools.product(*map(range, shape)):
            value = get_value(seq, get_seq_index(index))
            if value == seq_unspecified:
                value = unspecified
            if value == unspecified:
                continue
            for i in range(len(shape)):
                indices[i].append(index[i])
            values.append(value)
        data = Buffer(len(values), 0, values, unspecified=unspecified)
        indices = Strided.fromobject(indices)
        return cls(shape, indices, data)

    def __init__(self, shape:tuple, indices:Strided, data:Buffer):
        assert isinstance(shape, tuple)
        assert isinstance(indices, Strided)
        assert isinstance(data, Buffer)
        self.shape = shape
        self.indices = indices
        self.data = data
        self.size = indices.shape[1]
        assert self.size == self.data.size

    def _get_object_data(self):
        return self.shape, self.indices, self.data

    def indices_and_data_position(self):
        for data_pos in range(self.indices.shape[1]):
            index = tuple(self.indices[j, data_pos] for j in range(self.indices.shape[0]))
            yield index, data_pos

    def get_data_position(self, index):
        return get_coo_position(index, self.shape, self.indices)

    def get_slice(self, shape, strides, offset):
        values = []
        indices = [[] for d in shape]
        for slice_index, data_pos in sorted(self.slice_indices_and_data_position(shape, strides, offset)):
            for i in range(len(shape)):
                indices[i].append(slice_index[i])
            values.append(self.data[data_pos])
        indices_shape = (len(shape), len(indices[0]))
        indices_buffer = Buffer(product(indices_shape), 0, sum(indices, []))
        indices = Strided(indices_shape, make_strides(indices_shape), indices_buffer)

        data = Buffer(len(values), 0, values, unspecified=self.unspecified)
        return COO(shape, indices, data)        


class FlatCOO(Storage):
    """Represents a multidimensional array using COO storage format with indices flattened.
    """
    @classmethod
    def fromobject(cls, seq, shape=None, get_seq_index=None, unspecified=DEFAULT):
        if isinstance(seq, FlatCOO) and unspecified is DEFAULT and shape is None and get_seq_index is None:
            return seq
        if isinstance(seq, COO) and shape is None and get_seq_index is None and unspecified is DEFAULT:
            findices = []
            for index, data_pos in seq.indices_and_data_position():
                findices.append(get_position(index, seq.shape, seq.strides))
            findices= Buffer(len(findices), 0, findices)
            return cls(seq.shape, findices, seq.data.copy())
        # TODO: seq is CSR
        seq_unspecified = get_unspecified(seq)
        if unspecified is DEFAULT:
            unspecified = seq_unspecified
        if shape is None:
            shape = get_shape(seq)
        if get_seq_index is None:
            get_seq_index = lambda index: index
        strides = make_strides(shape)
        findices = []
        values = []
        for index in itertools.product(*map(range, shape)):
            value = get_value(seq, get_seq_index(index))
            if value == seq_unspecified:
                value = unspecified
            if value == unspecified:
                continue
            findex = get_position(index, shape, strides)
            findices.append(findex)
            values.append(value)
        data = Buffer(len(values), 0, values, unspecified=unspecified)
        findices= Buffer(len(findices), 0, findices)
        return cls(shape, findices, data)

    def __init__(self, shape:tuple, findices:Buffer, data:Buffer):
        assert isinstance(shape, tuple)
        assert isinstance(findices, Buffer)
        assert isinstance(data, Buffer)
        self.shape = shape
        self.findices = findices
        self.data = data
        self.size = findices.size
        assert self.size == self.data.size

    def _get_object_data(self):
        return self.shape, self.findices, self.data

    def indices_and_data_position(self):
        for data_pos in range(self.size):
            findex = self.findices[data_pos]
            index = get_index(findex, self.shape, self.strides)
            yield index, data_pos

    def get_data_position(self, index):
        findex = get_position(index, self.shape, self.strides)
        return get_flatcoo_position(findex, self.shape, self.findices)

    def get_slice(self, shape, strides, offset):
        findices = []
        values = []
        slice_strides = make_strides(shape)
        for slice_index, data_pos in sorted(self.slice_indices_and_data_position(shape, strides, offset)):
            findex = get_position(slice_index, shape, slice_strides)
            findices.append(findex)
            values.append(self.data[data_pos])
        data = Buffer(len(values), 0, values, unspecified=self.unspecified)
        findices = Buffer(len(findices), 0, findices)
        return FlatCOO(shape, findices, data)


class CRS(Storage):
    """Represents a two-dimensional array using CRS format.
    """

    @classmethod
    def fromobject(cls, seq, shape=None, get_seq_index=None, unspecified=DEFAULT):
        if isinstance(seq, COO) and shape is None and get_seq_index is None and unspecified is DEFAULT:
            crow_indices = Buffer.fromobject(compress_indices(seq.indices[0], seq.shape[0]))
            col_indices = Buffer.fromobject(seq.indices[1].copy())
            return CRS(seq.shape, crow_indices, col_indices, seq.data.copy())
        # TODO: seq is FlatCOO
        return COO.fromobject(seq, shape=shape, get_seq_index=get_seq_index, unspecified=unspecified).as_CRS()

    def __init__(self, shape, crow_indices: Buffer, col_indices: Buffer, data: Buffer):
        """
        Parameters
        ----------
        shape : tuple
          The dimensions of a 2-D array
        crow_indices : Buffer
          Compressed row indices as 1-D sequence
        col_indices : Buffer
          An array of column indices
        data : Buffer
          Array element values
        """
        assert len(shape) == 2
        assert isinstance(crow_indices, Buffer)
        assert isinstance(col_indices, Buffer)
        assert isinstance(data, Buffer)
        assert len(crow_indices) == shape[0] + 1
        assert len(col_indices) == len(data)

        self.shape = shape
        self.crow_indices = crow_indices
        self.col_indices = col_indices
        self.data = data
        self.size = data.size

    def _get_object_data(self):
        return self.shape, self.crow_indices, self.col_indices, self.data

    def indices_and_data_position(self):
        for row in range(self.shape[0]):
            for data_pos in range(self.crow_indices[row], self.crow_indices[row+1]):
                col = self.col_indices[data_pos]
                yield (row, col), data_pos

    def get_data_position(self, index):
        return get_crs_position(index, self.shape, self.crow_indices, self.col_indices)

    def get_slice(self, shape, strides, offset):
        if len(shape) == 1:
            return self.as_COO().get_slice(shape, strides, offset)

        assert len(shape) == 2, repr((shape, strides, offset))
        values = []
        row_indices = []
        col_indices = []
        for slice_index, data_pos in sorted(self.slice_indices_and_data_position(shape, strides, offset)):
            row, col = slice_index
            row_indices.append(row)
            col_indices.append(col)
            values.append(self.data[data_pos])
        data = Buffer(len(values), 0, values, unspecified=self.unspecified)
        crow_indices = Buffer.fromobject(compress_indices(row_indices, shape[0]))
        col_indices = Buffer.fromobject(col_indices)
        return CRS(shape, crow_indices, col_indices, data)        


def make_reduction(shape, rdimensions):
    """
    Parameters
    ----------
    shape : tuple
      Array shape
    rdimensions : tuple
      Dimensions mapping to reduced dimensions

    Returns
    -------
    rshape, rshapes, rstrides
    """
    rshape = []
    rshapes = []
    rstrides = []
    for rdims in rdimensions:
        reduction_shape = tuple(map(shape.__getitem__, rdims))
        reduction_stride = tuple(product(reduction_shape[i + 1:]) for i in range(len(reduction_shape)))
        assert make_strides(reduction_shape) == reduction_stride, (make_strides(reduction_shape), reduction_stride)
        rshape.append(product(reduction_shape))
        rshapes.append(reduction_shape)
        rstrides.append(reduction_stride)
    return tuple(rshape), tuple(rshapes), tuple(rstrides)

    
    
class GCS(Storage):
    """Generalized Compressed Storage format uses dimensionality reduction
    of multi-dimensional arrays so that N-dimensional array can be
    stored using the storage of an M-dimensional array where M <
    N. The compression effect is achieved when using CRS (M=2) with
    appropiate dimensions permutation.
    """

    @classmethod
    def fromobject(cls, seq, shape=None, dimensions=None, partitioning=None, storage_class=None, get_seq_index=None, unspecified=DEFAULT):
        """Parameters
        ----------
        seq : sequence
          Any sequence object that is converted to a GCS storage object

        dimensions : tuple
          A permutation of dimension indices. Describes swapping axes
          before dimensionality reduction.

        partitioning: tuple
          Indices that partition dimensions. Describes the mapping of
          dimensions to reduced dimensions.

        storage_class: type
          Storage class of dimension reduced storage object

        get_seq_index: callable
          A function that maps storage index to sequence index. Use internally.
        """
        if shape is None:
            shape = get_shape(seq)

        dimensionality = len(shape)

        if storage_class is None:
            storage_class = Strided
            
        if dimensions is None:
            dimensions = tuple(range(dimensionality))

        if partitioning is None:
            if storage_class is Buffer or dimensionality == 1:
                # M has to be 1
                partitioning = ()
            else:
                # default M is 2
                partitioning = (dimensionality-1,)
        elif isinstance(partitioning, int):
            partitioning = (partitioning,)

        if partitioning:
            assert partitioning == tuple(sorted(partitioning))
            assert min(partitioning) > 0
            assert max(partitioning) < dimensionality

        if get_seq_index is None:
            get_seq_index = lambda index: index

        reduction_dimensionality = len(partitioning) + 1

        assert reduction_dimensionality <= dimensionality
        
        partition_positions = (0,) + partitioning + (dimensionality,)
        rdimensions = []
        for k in range(reduction_dimensionality):
            rdims = dimensions[partition_positions[k]:partition_positions[k+1]]
            assert rdims not in rdimensions
            rdimensions.append(rdims)

        roffset = (0,) * reduction_dimensionality
        rshape, rshapes, rstrides = make_reduction(shape, rdimensions)

        storage_get_seq_index = (lambda rindex, rshapes=rshapes,
                                 rdimensions=rdimensions,
                                 rstrides=rstrides,
                                 roffset=roffset,
                                 get_seq_index=get_seq_index: (
                                     get_seq_index(
                                         promote_index(rshapes, rdimensions,
                                                       rstrides, roffset, rindex))))

        options = dict(
            shape=rshape,
            get_seq_index=storage_get_seq_index,
            unspecified=unspecified
        )
        storage = storage_class.fromobject(seq, **options)
        assert len(storage.shape) == reduction_dimensionality, (len(storage.shape), reduction_dimensionality)
        return cls(shape, tuple(rdimensions), tuple(rstrides), tuple(roffset), storage)

    def __init__(self,
                 shape: tuple,
                 rdimensions: tuple,
                 rstrides: tuple,
                 roffset: tuple,
                 data: Storage):
        assert set.union(*map(set, rdimensions)) == set(range(len(shape))), (set.union(*map(set, rdimensions)), set(range(len(shape))))
        
        rshape, rshapes, _rstrides = make_reduction(shape, rdimensions)
        assert isinstance(shape, tuple)
        assert isinstance(rdimensions, tuple)
        assert isinstance(rshapes, tuple)
        assert isinstance(rstrides, tuple)
        assert isinstance(roffset, tuple)
        assert isinstance(data, Storage)
        #assert rstrides == tuple(make_strides(rshape) for rshape in rshapes)
        self.shape = shape
        self.rdimensions = rdimensions
        self.rshapes = rshapes
        self.rstrides = rstrides 
        self.roffset = roffset
        self.data = data
        self.size = data.size

    def _get_object_data(self):
        return self.shape, self.rdimensions, self.rstrides, self.roffset, self.data

    def indices_and_data_position(self):
        for index in itertools.product(*map(range, self.shape)):
            rindex = reduce_index(self.rshapes, self.rdimensions, self.rstrides, self.roffset, index)
            # Notice that rindex ordering does not match the ordering of self.data values in general!
            yield index, rindex

    def get_data_position(self, index):
        return reduce_index(self.rshapes, self.rdimensions, self.rstrides, self.roffset, index)

    def __getitem__(self, index):
        index = slice_tools.normalize_index(index, self.shape)
        if is_element_index(index, self.shape):
            data_pos = self.get_data_position(index)
            if data_pos is None:
                return self.unspecified
            return self.data[data_pos]
        index = slice_tools.normalize_index(index, self.shape)
        gshape = slice_tools.shape_strided(index, self.shape)
        axis_to_be_removed = set(i for i in range(len(self.shape)) if isinstance(index[i], int))

        axis_map = []  # describes re-indexing of axis when some axes are fixed.
        for i in range(len(self.shape)):
            if i in axis_to_be_removed:
                axis_map.append(None)
            else:
                axis_map.append(i - sum(1 for j in range(i) if j in axis_to_be_removed))

        rshapes = ()
        roffset = ()
        rstrides = ()
        dindex = ()
        rdimensions = ()
        for k in range(len(self.rdimensions)):
            kslice = tuple(index[i] for i in self.rdimensions[k])
            kshape, kstrides, koffset = slice_tools.index_strided(kslice, self.rshapes[k], self.rstrides[k])
            if kshape:
                rshapes += (kshape,)
                roffset += (self.roffset[k] + koffset,)
                rstrides += (kstrides,)
                dindex += (None,)
            else:
                dindex += (koffset,)
            t = tuple(axis_map[i] for i in self.rdimensions[k] if i not in axis_to_be_removed)
            if t:
                rdimensions += (t,)

        # Notice that slicing of GCS does not involve access to
        # storage data when the sliced array has the same
        # dimensionality (holds also for COO and CRS use cases) or
        # when the data storage uses strided storage format.
        data = self.data[dindex] if dindex else self.data
        return GCS(gshape, rdimensions, rstrides, roffset, data)
