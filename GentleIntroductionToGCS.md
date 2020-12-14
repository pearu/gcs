<!--watch-latex-md

This document is processed by watch_latex_md.py program, see

  https://github.com/Quansight/pearu-sandbox/latex_in_markdown/

You can edit this document as you wish. You can also edit the LaTeX
data in img elements, but only the content of `latex-data`:

  1. To automatically update the LaTeX rendering in img element, edit
     the file while watch_latex_md.py is running.

  2. Never change the beginning (`<img latex-data="...`) and the end
     (`...alt="latex">`) parts of the LaTeX img elements as these are
     used by the watch_latex_md.py script.

  3. Changes to other parts of the LaTeX img elements will be
     overwritten.

Enjoy LaTeXing!

watch-latex-md:no-force-rerender
-->


# GCS - Generalized Compressed Storage of multi-dimensional arrays

|            |                 |
| ---------- | --------------- |
| Authors    | Pearu Peterson  |
| Created    | 2020-12-13      |

## Definitions

An array is a structured collection of array elements. An array
element is a pair of its associated index and a value.

The element index is unique to each array element and it enables
locating the corresponding element in the array structure. It is
convinient to use tuples of integers as element indices such that the
length of the tuple defines the dimensionality of the array and each
item in the tuple is within an allowed range of integer values. The
lenghts of all ranges define the shape of the array.

The element value is usually a numerical value from some field
(integers, real numbers, complex numbers, etc) as we want to define
array operations that involve arithmetic operations with array
elements. We also assume that the values of all array elements are
from the same field --- this is manifested by defining a single value
type for all array elements.

For example, consider a 2-dimensional array with 6 elements that have
the following indices and values:

| Indices | Values
| ------- | ------
| `(0, 0)`  | `1`
| `(0, 1)`  | `2`
| `(0, 2)`  | `3`
| `(1, 0)`  | `4`
| `(1, 1)`  | `5`
| `(1, 2)`  | `6`

The shape of this array is `(2, 3)`.

To visualize arrays, in particular, the 2-dimensional arrays, we shall
use the table represenation of arrays where the row indices correspond
to the first index item and the column indices to the second index
item. For instance, the 2-dimensional array above has the following
table representation:
```
  1 2 3
  4 5 6
```
where the two rows correspond to the first index values (`0` or `1`) and
the three columns to the second index values (`0` or `1` or `2`).

## Array storage formats


There exists many formats that can be used for storing array elements
in some storage device, say, computer memory.

### Strided memory layout


The most efficient general storage scheme is strided storage format
where only the values of array elements are stored while the
corresponding elemenent indices can be computed from the memory
address of the array element, and vice-versa, from the index of an
array element one can compute the location of the array element value
in memory. To enable these computations, one needs to define a tuple
of integer valued strides. Then the memory location of the value of an
array element with index tuple `indices` is `offset + sum(strides[i] *
indices[i], i=0,...,N-1)` where `N` denotes the dimensionality of the
array and `offset` is an offset from a memory address.

For example, if the strides tuple is `(3, 1)` and offset is `0` then
the array in previous example would have the following indices,
values, and memory locations:

|Indices | Values | Memory location
|--------|--------|----------------
| `(0, 0)`  | `1`      | `0`
| `(0, 1)`  | `2`      | `1`
| `(0, 2)`  | `3`      | `2`
| `(1, 0)`  | `4`      | `3`
| `(1, 1)`  | `5`      | `4`
| `(1, 2)`  | `6`      | `5`

and the values would be layout in memory in the following order:
```
1 2 3 4 5 6
```

Other choices of strides tuple are possible, for instance, take
strides=(-1, -2), offset=5, and then we have

|Indices | Values | Memory location
|------- |--------|----------------
| `(0, 0)`  | `1`      | `5`
| `(0, 1)`  | `2`      | `3`
| `(0, 2)`  | `3`      | `1`
| `(1, 0)`  | `4`      | `4`
| `(1, 1)`  | `5`      | `2`
| `(1, 2)`  | `6`      | `0`

with the following memory layout for values:
```
  6 3 5 2 4 1
```

### Coordinate storage format (COO)

Notice that in strided storage format the values of *all* array
elements are stored in computer memory. In some occasions, the values
of only specific elements need to be stored. For instance, when most
of the values of array elements are known to be equal, say, to zero,
then a considreable storage saving can be achieved if only elements
with non-zero values are stored. The so-called coordinate storage
format (COO) allows storing only specific elements but this requires
that both the indices and valus of specified elements are explicitly
stored in computer memory. According to COO storage format, the
indicies and values of specified elements are stored in two arrays,
indices and values, that use, say, the strided storage format as
described above.

For example, consider a 2-dimensional array with shape `(2, 3)` that
has the following specified elements:

| Indices | Values
| --------|-------
| `(0, 1)`| `1`
| `(1, 0)`| `2`
| `(1, 2)`| `3`

This array in the table representation is as follows
```
  X 1 X
  2 X 3
```
where `X` denotes the placeholder for unspecified elements.

This array in COO storage format is represented by arrays of indices
and values as follows (using table representation for simplicity):
```
indices:

  0 1 1
  1 0 2

values:

  1 2 3
```

COO storage format can be easily generalized to use it for storing the
specified elements of a multi-dimensional array: the shape of indices
array would be `(N, NSE)` where `N` is the dimensionality of the array
and `NSE` is the number of specified elements.


### Compressed Row/Column Storage formats (CRS/CCS)


(Compressed row/column storage formats are also known as compressed
sparse row/column storage formats CSR/CSC. Here we avoid the usage of
"sparse" as it is ambiguous adjective and the discussed array formats
are sparsity agnostic.)

While COO storage format is a popular format for storing sparse arrays
because of its simplicity, it is not the most efficient storage format
from the view point of processing sparse arrays, say, for matrix
multiplication operations. The so-called compressed row storage (CRS)
format is a certain specialization of the COO storage format where the
row indices are stored in a special way that makes algorithms using
loops over row indices particularly processor efficent. The CRS format
also saves some storage space because row indices are stored in a
compressed way.

As an example, let's consider a 2-dimensional array with a shape `(4,
5)` that has the following table representation:
```
  X X 1 X 2
  3 X X 4 X
  5 X 6 7 X
  X X X 8 9
```
In the COO storage format, this array reads
```
indices:

  0 0 1 1 2 2 2 3 3
  2 4 0 3 0 2 3 3 4

values:

  1 2 3 4 5 6 7 8 9
```

In the CRS storage format, the first row of indices (corresponding to
row indices of the array elements) is rewritten to hold the
cummulative number of column elements for each row: the first and
second rows contain 2 elements, the third row contains 3 elements and
the last again the two elements. So, the cummulative number of column
elements for all rows is `2`, `2+2=4`, `2+2+3=7`, `2+2+3+2=9`,
respectively. In the CRS format, the array reads:
```
compressed row indices, crow_indices:

  0 2 4 7 9

column indices, col_indices:

  2 4 0 3 0 2 3 3 4

values:

  1 2 3 4 5 6 7 8 9
```

Notice that the first item of `crow_indices` corresponds to the index of
first row that contains specified elements and the last item in
`crow_indices` is always equal to the number of specified elements, that
is, the size of col_indices and values arrays.

Analogously to CRS, one can define compressed column storage format
(CCS) where the array column indices are rewritten in compressed
form. For example, the sample array above in the CCS format reads:
```
row indices, row_indices:

  1 2 0 2 1 2 3 0 3

compressed column indinces, ccol_indices:

  0 2 2 4 7 9

values:

  3 5 1 6 4 7 8 2 9
```

Similarly to COO storage format, the CRS/CCS formats can be
generalized for multi-dimensional arrays, however, many different
generalizations have been proposed because for `N`-dimensional array
there is a freedom of selecting which dimension indices will be
rewritten in compressed way. Recall, that for `N=2`, we have two
choices leading to CRS and CCS storage formats. For `N>2` cases, one
can define more than two different storage formats that are
parameterized by the choice of compressed dimension, or more generally
speaking, combinations of compressed dimensions.


## Generalized Comporessed Storage format (GCS)

For generalizing CRS/CCS formats to storing N-dimensional arrays, we
are going to use and generalize the basic idea behind the GCRS/GCCS
formats introduced in "Efficient storage scheme for n-dimensional
sparse array: GCRS/GCCS"
(https://ieeexplore.ieee.org/document/7237032): use dimensionality
reduction to map the N-dimensional indicies to 2-dimensional indices
and apply the standard CRS format for storing the elements in the
dimensionality reduced 2-dimensional array.  The authors of the paper
described this method for just one possible reduction scheme that
efficency may not be optimal for array operations in general (they
split the `N` dimensions to odd and even dimensions and then used
dimensionality reduction such that the odd and even dimensions map to
row and column dimensions, respectively) and having a choice of mixing
dimensions would allow more optimal data locality for array operations
that could enhance array processing performance.

### The basics of dimensionality reduction

Consider a `N`-dimensional array and the strided storage format. The
values of array elements are stored in a computer memory as a linear
chunk of data that could be considered as a 1-dimensional array by its
own. This view of an `N`-dimensional array is the dimensionality
reduction from `N` to `1`. The dimensionality reduction has reverse
operation, the dimensionality promotion, that maps 1-dimensional
indices (index) to `N`-dimensional indices. The `N-to-1` dimensionality
reduction and `1-to-N` dimensionality promotions are described by the
strides and shape tuples.

If `indices` is a `N`-tuple of integers and `index` is an integer, then
the `N-to-1` dimensionality reduction is as a map of `N`-dimensional
indices to 1-dimensional index:
```
  index = sum(indices[i] * strides[i], i=0, ..., N-1)
```

The `1-to-N` dimensionality promotion is defined as a map of a
1-dimensional index to `N`-dimensional indices:
```
  indices[i] = index // strides[i]  mod shape[i],   i=0, ..., N-1.
```

This formalism can be easily generalized to N-to-M dimensionality
reduction and `M-to-N` dimensionality promotion for arbitrary N and M
such that `M < N`. In this document we consider only `N-to-2`
dimensionality reductions and its reverse for the sake of simplicity
in introducing the GCS format, however, the tools in [gcs_basics.py](gcs_basics.py)
provide implementations for general `M-to-N` dimensions reductions.

There is a number of ways to perform the `N-to-2` dimensionality
reduction depending of the choice of mapping the subsets of `N`
dimensions to row or column indices. For example, consider a
3-dimensional array with shape `(2, 3, 8)` and its dimensionality
reduction to 2-dimensional array. We have the following choices:

1. map the first two dimensions to reduced row dimension and the last
   dimension to reduced column dimension, the dimensionality reduced array will have shape `(2*3, 8)`;

2. map the first dimension to reduced row dimension and the rest
   dimensions to reduced column dimension, the dimensionality reduced array will have shape `(2, 3*8)`;

3. map the last dimension to reduced row dimension and the first two
   dimensions to reduced column dimension, the dimensionality reduced array will have shape `(8, 2*3)`;

4. map the last dimension to reduced row dimension and the first two
   dimensions, swapped, to reduced column dimension, the
   dimensionality reduced array will have shape `(8, 3*2)`;

etc. In total, there are 12 variatons to reduce the dimensionality of
a 3-dimensional array to a 2-dimensional array. To describe the
dimensions mappings, we will introduce a tuple of dimensions
permutations and a tuple of dimensions partitioning.

The tuple of dimensions permutations describes the swapping of axes
before performing the dimensions reduction. It is a `N`-tuple of
integers containing any permutation of the tuple `(0, 1, ..., N-1)`.

In general, the tuple of dimensions partitioning is a (M-1)-tuple of
positive ordered integers less than N-1, that defines the partitioning
of the dimensions permuations tuple into M parts. For `M=2` case, the
parts correspond to the reduced row and reduced column dimensions and
the tuple of dimensions partitioning contains only one integer value.

For example, define `dimensions=(2, 4, 1, 3, 0)` as the dimensions
permuation of a 5-dimensional array and let `partitioning=(3, )` be
the dimensions partitioning tuple. Then the 3rd, 5th, and 2nd
dimensions are mapped to reduced row dimension and the 4th, and 1st
dimensions are mapped to reduced column dimension:
```
  row_dimensions = (2, 4, 1)
  col_dimensions = (3, 0)
```

For N-to-2 dimensionality reductions, there exist two reduction stride
tuples that define N-to-1 dimensionality reductions of dimensions in
`row_dimensions` and `col_dimensions`, respecitively:

```
  row_strides[i] = product(shape[row_dimensions[k]], k=1, ..., Nr - 1 - i),  i=0, ..., Nr - 1
  col_strides[j] = product(shape[col_dimensions[k]], k=1, ..., Nc - 1 - j),  j=0, ..., Nc - 1
```
where `Nr = len(row_dimensions)` and `Nc = len(col_dimensions)`.

For example, let `shape=(2, 3, 4, 5, 6)`, `dimensions=(2, 4, 1, 3,
0)`, and `partitioning=(3,)`, then we have

```
  row_strides = [6*3, 3, 1] = [18, 3, 1]
  col_strides = [2, 1]
```

Now, we have everything needed for mapping N-dimensional indices to
GCS row and column indices. Given a tuple of indices, we can compute
```
  row_index = sum(row_strides[i] * indices[row_dimensions[i]], i=0, ..., Nr - 1)
  col_index = sum(col_strides[j] * indices[col_dimensions[j]], j=0, ..., Nc - 1)
```

and vice-versa, we can compute N-dimensional indices from the given GCS row and column indices:

```
  indices[row_dimensions[i]] = (row_index // row_strides[i])  % shape[row_dimensions[i]]
  indices[col_dimensions[j]] = (col_index // col_strides[j])  % shape[col_dimensions[j]]
```

where `i=0, ..., Nr - 1` and `j=0, ..., Nc - 1`.

For example, let's consider a 3-dimensional array with shape `(2, 3,
4)` with the following elements specified:

| Indices   | Values
| ----------|-------
| `(0, 0, 1)` | `1`
| `(0, 0, 2)` | `2`
| `(0, 0, 3)` | `3`
| `(0, 2, 1)` | `4`
| `(1, 0, 0)` | `5`
| `(1, 0, 3)` | `6`
| `(1, 2, 0)` | `7`
| `(1, 2, 2)` | `8`
| `(1, 2, 3)` | `9`

To reduce its dimensionality to 2, let's use dimensions permutation
tuple `(0, 1, 2)` (read: no dimensions are swapped), and partitioning
tuple `(2,)` (read: the first two dimensions are mapped to reduced row
dimension and the last dimension is mapped to reduced column
dimension). Then we have the following mapping of indices:

| Indices     | Reduced row index | Reduced column index
| ----------  |-------------------|---------------------
| `(0, 0, 1)` | `0`                 | `1`
| `(0, 0, 2)` | `0`                 | `2`
| `(0, 0, 3)` | `0`                 | `3`
| `(0, 2, 1)` | `2`                 | `1`
| `(1, 0, 0)` | `3`                 | `0`
| `(1, 0, 3)` | `3`                 | `3`
| `(1, 2, 0)` | `5`                 | `0`
| `(1, 2, 2)` | `5`                 | `2`
| `(1, 2, 3)` | `5`                 | `3`

This represents the array of dimensionality reduction:
```
X 1 2 3
X X X X
X 4 X X
5 X X 6
X X X X
7 X 8 9
```
that could be stored in COO format:
```
indices:

  0 0 0 2 3 3 5 5 5
  1 2 3 1 0 3 0 2 3

values:

  1 2 3 4 5 6 7 8 9
```
or in CRS format:
```
crow_indices:

  0 3 3 4 6 6 9

col_indices:

  1 2 3 1 0 3 0 2 3

values:

  1 2 3 4 5 6 7 8 9
```
For `partitioning=(1,)` (the first dimension is mapped to the reduced
row dimension and the last two dimensions are mapped to the reduced
column dimension), the array of dimensionality reduction is:
```
X 1 2 3 X X X X X 4 X X
5 X X 6 X X X X 7 X 8 9

or in CRS format:

crow_indices:

  0 4 9

col_indices:

  1 2 3 9 0 3 8 10 11

values:

  1 2 3 4 5 6 7 8 9
```
For `dimensions=(2, 1, 0)`, `partitioning=(1,)` (the last dimension is mapped to the reduced
row dimension and the first two dimensions swapped are mapped to the reduced
column dimension), the array of dimensionality reduction is:
```
X 5 X X X 7
1 X X X 4 X
2 X X X X 8
3 6 X X X 9
```
or in CRS format:
```
crow_indices:

  0 2 4 6 9

col_indices:

  1 5 0 4 0 5 0 1 5

values:

  5 7 1 4 2 8 3 6 9
```

Notice that the choice of dimensions mapping can result in different
memory layouts for arrays of dimensionality reductions that can be
advantageous or disadvantageous for further processing of
dimensionality reduced arrays. Also, the required storage size of
array data depends on the choice of dimensions mapping.

The following table illustrates all possible mappings for a
3-dimensional array with shape `(2, 3, 4)` (the array element indices
are encoded into the corresponding element values, for instance, the
value `112` indicates that the corresponding element index is `(1, 1,
2)`):



| Row / Column dimensions   | 2-D reduction array                                 |
| --- | --- | 
| `[0]`      / `[1, 2]`     |  `000 001 002 003 010 011 012 013 020 021 022 023`  |
|                           |  `100 101 102 103 110 111 112 113 120 121 122 123`  |
| --- | --- | 
| `[0]`      / `[2, 1]`     |  `000 010 020 001 011 021 002 012 022 003 013 023`  |
|                           |  `100 110 120 101 111 121 102 112 122 103 113 123`  |
| --- | --- | 
| `[1]`      / `[0, 2]`     |  `000 001 002 003 100 101 102 103`                  |
|                           |  `010 011 012 013 110 111 112 113`                  |
|                           |  `020 021 022 023 120 121 122 123`                  |
| --- | --- | 
| `[1]`      / `[2, 0]`     |  `000 100 001 101 002 102 003 103`                  |
|                           |  `010 110 011 111 012 112 013 113`                  |
|                           |  `020 120 021 121 022 122 023 123`                  |
| --- | --- | 
| `[2]`      / `[0, 1]`     |  `000 010 020 100 110 120`                          |
|                           |  `001 011 021 101 111 121`                          |
|                           |  `002 012 022 102 112 122`                          |
|                           |  `003 013 023 103 113 123`                          |
| --- | --- | 
| `[2]`      / `[1, 0]`     |  `000 100 010 110 020 120`                          |
|                           |  `001 101 011 111 021 121`                          |
|                           |  `002 102 012 112 022 122`                          |
|                           |  `003 103 013 113 023 123`                          |
| --- | --- | 
| `[0, 1]`   / `[2]`        |  `000 001 002 003`                                  |
|                           |  `010 011 012 013`                                  |
|                           |  `020 021 022 023`                                  |
|                           |  `100 101 102 103`                                  |
|                           |  `110 111 112 113`                                  |
|                           |  `120 121 122 123`                                  |
| --- | --- | 
| `[0, 2]`   / `[1]`        |  `000 010 020`                                      |
|                           |  `001 011 021`                                      |
|                           |  `002 012 022`                                      |
|                           |  `003 013 023`                                      |
|                           |  `100 110 120`                                      |
|                           |  `101 111 121`                                      |
|                           |  `102 112 122`                                      |
|                           |  `103 113 123`                                      |
| --- | --- | 
| `[1, 0]`   / `[2]`        |  `000 001 002 003`                                  |
|                           |  `100 101 102 103`                                  |
|                           |  `010 011 012 013`                                  |
|                           |  `110 111 112 113`                                  |
|                           |  `020 021 022 023`                                  |
|                           |  `120 121 122 123`                                  |
| --- | --- | 
| `[1, 2]`   / `[0]`        |  `000 100`                                          |
|                           |  `001 101`                                          |
|                           |  `002 102`                                          |
|                           |  `003 103`                                          |
|                           |  `010 110`                                          |
|                           |  `011 111`                                          |
|                           |  `012 112`                                          |
|                           |  `013 113`                                          |
|                           |  `020 120`                                          |
|                           |  `021 121`                                          |
|                           |  `022 122`                                          |
|                           |  `023 123`                                          |
| --- | --- | 
| `[2, 0]`   / `[1]`        |  `000 010 020`                                      |
|                           |  `100 110 120`                                      |
|                           |  `001 011 021`                                      |
|                           |  `101 111 121`                                      |
|                           |  `002 012 022`                                      |
|                           |  `102 112 122`                                      |
|                           |  `003 013 023`                                      |
|                           |  `103 113 123`                                      |
| --- | --- | 
| `[2, 1]`   / `[0]`        |  `000 100`                                          |
|                           |  `010 110`                                          |
|                           |  `020 120`                                          |
|                           |  `001 101`                                          |
|                           |  `011 111`                                          |
|                           |  `021 121`                                          |
|                           |  `002 102`                                          |
|                           |  `012 112`                                          |
|                           |  `022 122`                                          |
|                           |  `003 103`                                          |
|                           |  `013 113`                                          |
|                           |  `023 123`                                          |


<!--EOF-->
