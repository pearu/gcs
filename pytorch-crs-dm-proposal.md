# CRS/CCS and DM storage formats proposal - executive summary

|            |                 |
| ---------- | --------------- |
| Authors    | Pearu Peterson  |
| Created    | 2021-01-15      |

Currently, PyTorch implements the following multi-dimensional array
storage formats:

- Strided: an array is defined by shape, strides, and 1D values
  (PyTorch internal buffer).
- COO: an array is defined by shape, indices (PyTorch 2D Strided
  tensor), and values (PyTorch 1D Strided tensor).

Here we do not consider hybrid COO tensors for simplicity.

For matrix multiplication of sparse tensors, the COO storage format is
suboptimal and more memory/processor efficient storage formats are
needed, especially those that allow parallel processing of array data
with minimal cache misses and in streaming fashion.

We propose implementing the following new storage formats in PyTorch:

- CRS/CCS: a 2D array is defined by shape, compressed row/column
  indices (PyTorch 1D tensor), column/row indices (PyTorch 1D tensor),
  and values (PyTorch 1D tensor).

- DM: an array is defined by shape, dimensions map, strides map,
  offset, and storage array (an arbitrary PyTorch tensor).

The DM (Dimensions Mapping) storage format is a new concept that
constitutes of dimensionality reduction and promotion of the storage
array as defined by dimensions and strides maps and offset parameters
(see the Python prototype below for detailed definitions). The DM
storage array is a PyTorch tensor that can use any of the existing
storage formats (Strided, COO) or any of the proposed new storage
formats: CRS/CCS and DM. The dimensionality of the storage array is
not greater than the dimensionality of the DM array.

While CRS/CCS formats support more performant Linear Algebra
operations with 2D sparse arrays (see the benchmark results in
[PyTorch PR 44190](https://github.com/pytorch/pytorch/pull/44190)
description), the DM format provides a method of generalizing the 2D
CRS/CCS formatted arrays to multidimensional arrays. In addition, the
DM format allows several new perspectives to multidimensional array
representations:

- no-copy slicing and axis swapping for COO and CRS/CSS tensors.

- hierarchical storage format: DM tensor can use another DM tensor as
  the storage array.

- extending storage formats with restricted dimensioniality (e.g. the
  CRS format is essentially for 2D arrays only) to arbitrary
  dimensionality.

A pure Python prototype of DM storage format is provided in [GCS
package](https://github.com/pearu/gcs/gcs) (the DM storage format is
provided under the name GCS) where the dimensions mapping and slicing
of DM arrays are implemented and demonstrated on wrapping of storage
arrays using various storage formats like Strided, COO, FlatCOO, CRS,
and CCS. See also [Gentle Introduction To
GCS](GentleIntroductionToGCS.md) that exemplifies the dimensions
mapping concepts in more detail.

Execution plan:

- Implement CRS storage format, the implementation exists in the form
  of [PyTorch PR 44190](https://github.com/pytorch/pytorch/pull/44190)
  but it requires some clean up.
- Implement DM storage format in storage array format agnostic way.
- Implement CCS storage format as a follow up to CRS.

The PyTorch PR 44190 is kept for reference but will be retired
eventually.

## Notations

COO format is Coordinate storage format.

Compressed Row Storage (CRS) format is the same as Compresses Sparse
Row (CSR) storage format.

Compressed Column Storage (CCS) format is the same as Compresses
Sparse Column (CSC) storage format.

DM is Dimensions Mapping storage format .
