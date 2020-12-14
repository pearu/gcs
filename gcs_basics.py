"""

"""
# Author: Pearu Peterson
# Created: December 2020

__all__ = ['create_reduction_metadata', 'reduction', 'promotion', 'compress_indices']



def product(seq):
    return seq[0] * product(seq[1:]) if seq else 1


def create_reduction_metadata(shape, dimensions, partitioning):
    """Create metadata for dimensionality reduction of a multidimensional array.

    Parameters
    ----------
    shape : tuple
      Shape of an array
    dimensions : tuple
      A permutation of dimension indices. Describes swapping axes
      before dimensionality reduction.

    partitioning: tuple
      Indices that partition dimensions. Describes the mapping of
      dimensions to reduced dimensions.

    Returns
    -------
    metadata : dict
      Various parameters that are used in computing dimensionality
      reductions.

    """
    dimensionality = len(shape)
    reduction_dimensionality = len(partitioning) + 1
    assert reduction_dimensionality <= dimensionality, (reduction_dimensionality, dimensionality, shape, reduction)
    partition_positions = (0,) + partitioning + (dimensionality,)
    reduction_dimensions = []
    reduction_shapes = []
    reduction_strides = []
    for k in range(reduction_dimensionality):
        rdimensions = dimensions[partition_positions[k]:partition_positions[k+1]]
        reduction_shape = tuple(map(shape.__getitem__, rdimensions))
        reduction_stride = tuple(product(reduction_shape[i + 1:]) for i in range(len(reduction_shape)))
        reduction_dimensions.append(rdimensions)
        reduction_shapes.append(reduction_shape)
        reduction_strides.append(reduction_stride)
    reduction_shape = tuple(map(product, reduction_shapes))
    return dict(
        dimensionality=dimensionality,
        shape=shape,
        reduction_dimensionality=reduction_dimensionality,
        reduction_shape=reduction_shape,
        reduction_dimensions=tuple(reduction_dimensions),
        reduction_shapes=tuple(reduction_shapes),
        reduction_strides=tuple(reduction_strides))


def reduction(metadata, indices):
    """Apply dimensionality reduction to indices.

    Returns reduced indices.
    """
    reduction_indices = []
    for k in range(len(metadata['reduction_shape'])):
        reduction_shape = metadata['reduction_shapes'][k]
        reduction_strides = metadata['reduction_strides'][k]
        reduction_dimensions = metadata['reduction_dimensions'][k]
        reduction_index = sum(reduction_strides[i] * indices[reduction_dimensions[i]] for i in range(len(reduction_shape)))
        reduction_indices.append(reduction_index)
    return tuple(reduction_indices)


def promotion(metadata, reduction_indices):
    """Apply dimensionality promotion to reduced indices. The inverse of
    reduction.

    Returns indices.
    """
    indices = [None] * metadata['dimensionality']
    for k in range(metadata['reduction_dimensionality']):
        reduction_shape = metadata['reduction_shapes'][k]
        reduction_strides = metadata['reduction_strides'][k]
        reduction_dimensions = metadata['reduction_dimensions'][k]
        for i in range(len(reduction_dimensions)):
            indices[reduction_dimensions[i]] = (reduction_indices[k] // reduction_strides[i]) % reduction_shape[i]
    return tuple(indices)


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
