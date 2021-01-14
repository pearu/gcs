import pytest
import itertools
from gcs.storage import Buffer, Strided, COO, CRS, GCS, FlatCOO


def test_Buffer():
    s = Buffer.fromobject([[1, 2, 3], [4, 5, 6]])
    s2 = Buffer(6, 0, [1, 2, 3, 4, 5, 6])
    assert s == s2
    for i in range(6):
        assert s[i] == i + 1
    s2[3] = 22
    assert s2[3] == 22
    assert s != s2
    assert s.size == 6
    assert s.shape == (6,)

    assert s[:2] == Buffer(2, 0, s.data)
    assert len(s[:2]) == 2
    assert s[:2].size == 2
    assert s[:2].shape == (2,)
    assert len(s[1:2]) == 1
    assert list(s[:2]) == s.data[:2]
    assert list(s[1:2]) == s.data[1:2]

    assert s[2:5][0] == 3
    assert s[2:5][1] == 4
    assert s[2:5][2] == 5
    assert s[2:5][3] == 6
    assert s[2:5][-1] == 2
    assert s[2:5][-2] == 1

    assert s[2:][:].equal([3, 4, 5, 6])
    assert s[2:][:].size == 4
    assert s[2:][-1:].equal([2, 3, 4, 5, 6])
    assert s[2:][-1:].size == 5
    assert s[2:4][-1:].equal([2, 3, 4])
    assert s[2:4][-2:].equal([1, 2, 3, 4])
    assert s[2:4][-2:2].equal([1, 2, 3, 4])
    assert s[2:4][-2:3].equal([1, 2, 3, 4, 5])
    assert s[2:4][-2:1].equal([1, 2, 3])


def test_Strided():
    s = Strided.fromobject([[1, 2, 3], [4, 5, 6]])
    s2 = Strided((2, 3), (3, 1), Buffer.fromobject([1, 2, 3, 4, 5, 6]))
    assert s == s2
    assert s[0, 0] == 1
    assert s[0, 1] == 2
    assert s[0, 2] == 3
    assert s[1, 0] == 4
    assert s[1, 1] == 5
    assert s[1, 2] == 6
    assert s.shape == (2, 3)

    s2[1, 1] = 22
    assert s2[1, 1] == 22
    assert s != s2

    assert s[0].equal([1, 2, 3])
    assert s[0].size == 3
    assert s[0].shape == (3, )
    assert s[1].equal([4, 5, 6])
    assert s[-1].equal([4, 5, 6])
    assert s[:, 0].equal([1, 4])
    assert s[:, 1].equal([2, 5])
    assert s[:, 2].equal([3, 6])
    assert s[:, -1].equal([3, 6])
    assert s[:, -2].equal([2, 5])

    assert s[0:].equal([[1, 2, 3], [4, 5, 6]])
    assert s[1:].equal([[4, 5, 6]])
    assert s[::-1].equal([[4, 5, 6], [1, 2, 3]])
    assert s[-2::-1].equal([[1, 2, 3]])

    assert s[:, 0:].equal([[1, 2, 3], [4, 5, 6]])
    assert s[:, 1:].equal([[2, 3], [5, 6]])
    assert s[:, 2:].equal([[3], [6]])

    assert s[:, 0:2].equal([[1, 2], [4, 5]])
    assert s[:, :-1].equal([[1, 2], [4, 5]])
    assert s[:, 1:2].equal([[2], [5]])
    assert s[:, ::2].equal([[1, 3], [4, 6]])
    assert s[:, ::-1].equal([[3, 2, 1], [6, 5, 4]])
    assert s[:, ::-2].equal([[3, 1], [6, 4]])
    assert s[:, -2::-1].equal([[2, 1], [5, 4]])

    assert s.copy().equal(s)
    assert s.equal(s.copy())
    assert s == s.copy()
    assert s.copy(unspecified=0).equal(s)
    assert s != s.copy(unspecified=0)


def test_COO():
    s = COO.fromobject([[1, 2, 3], [4, 5, 6]])
    s2 = COO((2, 3), Strided.fromobject([[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]]),
             Buffer.fromobject([1, 2, 3, 4, 5, 6]))
    assert s == s2
    assert s.is_full()
    assert s[0, 0] == 1
    assert s[0, 1] == 2
    assert s[0, 2] == 3
    assert s[1, 0] == 4
    assert s[1, 1] == 5
    assert s[1, 2] == 6
    assert s.shape == (2, 3)

    assert s[0].equal([1, 2, 3])
    assert s[0].size == 3
    assert s[0].shape == (3, )
    assert s[1].equal([4, 5, 6])
    assert s[-1].equal([4, 5, 6])
    assert s[:, 0].equal([1, 4])
    assert s[:, 1].equal([2, 5])
    assert s[:, 2].equal([3, 6])
    assert s[:, -1].equal([3, 6])
    assert s[:, -2].equal([2, 5])

    assert s[0:].equal([[1, 2, 3], [4, 5, 6]])
    assert s[1:].equal([[4, 5, 6]])

    assert s[::-1].equal([[4, 5, 6], [1, 2, 3]])
    assert s[-2::-1].equal([[1, 2, 3]])

    assert s[:, 0:].equal([[1, 2, 3], [4, 5, 6]])
    assert s[:, 1:].equal([[2, 3], [5, 6]])
    assert s[:, 2:].equal([[3], [6]])

    assert s[:, 0:2].equal([[1, 2], [4, 5]])
    assert s[:, :-1].equal([[1, 2], [4, 5]])
    assert s[:, 1:2].equal([[2], [5]])
    assert s[:, ::2].equal([[1, 3], [4, 6]])
    assert s[:, ::-1].equal([[3, 2, 1], [6, 5, 4]])
    assert s[:, ::-2].equal([[3, 1], [6, 4]])
    assert s[:, -2::-1].equal([[2, 1], [5, 4]])

    s[1, 1] = 22
    assert s[1, 1] == 22
    assert s != s2

    s = COO.fromobject([[1, None, 3], [None, 5, None]])
    s2 = COO((2, 3), Strided.fromobject([[0, 0, 1], [0, 2, 1]]), Buffer.fromobject([1, 3, 5]))
    assert s == s2
    assert not s.is_full()

    assert s[0, 0] == 1
    assert s[0, 1] == s.unspecified
    assert s[0, 2] == 3
    assert s[1, 0] == s.unspecified
    assert s[1, 1] == 5
    assert s[1, 2] == s.unspecified

    s = COO.fromobject([[1, 0, 3], [0, 5, 0]], unspecified=0)
    s2 = COO((2, 3), Strided.fromobject([[0, 0, 1], [0, 2, 1]]),
             Buffer.fromobject([1, 3, 5], unspecified=0))
    assert s == s2
    assert not s.is_full()

    assert s[0, 0] == 1
    assert s[0, 1] == s.unspecified
    assert s[0, 2] == 3
    assert s[1, 0] == s.unspecified
    assert s[1, 1] == 5
    assert s[1, 2] == s.unspecified


def test_FlatCOO():
    s = FlatCOO.fromobject([[1, 2, 3], [4, 5, 6]])
    s2 = FlatCOO((2, 3), Buffer.fromobject([0, 1, 2, 3, 4, 5]),
                 Buffer.fromobject([1, 2, 3, 4, 5, 6]))
    assert s == s2
    assert s.is_full()
    assert s[0, 0] == 1
    assert s[0, 1] == 2
    assert s[0, 2] == 3
    assert s[1, 0] == 4
    assert s[1, 1] == 5
    assert s[1, 2] == 6
    assert s.shape == (2, 3)

    assert s[0].equal([1, 2, 3])
    assert s[0].size == 3
    assert s[0].shape == (3, )
    assert s[1].equal([4, 5, 6])
    assert s[-1].equal([4, 5, 6])
    assert s[:, 0].equal([1, 4])
    assert s[:, 1].equal([2, 5])
    assert s[:, 2].equal([3, 6])
    assert s[:, -1].equal([3, 6])
    assert s[:, -2].equal([2, 5])

    assert s[0:].equal([[1, 2, 3], [4, 5, 6]])
    assert s[1:].equal([[4, 5, 6]])

    assert s[::-1].equal([[4, 5, 6], [1, 2, 3]])
    assert s[-2::-1].equal([[1, 2, 3]])

    assert s[:, 0:].equal([[1, 2, 3], [4, 5, 6]])
    assert s[:, 1:].equal([[2, 3], [5, 6]])
    assert s[:, 2:].equal([[3], [6]])

    assert s[:, 0:2].equal([[1, 2], [4, 5]])
    assert s[:, :-1].equal([[1, 2], [4, 5]])
    assert s[:, 1:2].equal([[2], [5]])
    assert s[:, ::2].equal([[1, 3], [4, 6]])
    assert s[:, ::-1].equal([[3, 2, 1], [6, 5, 4]])
    assert s[:, ::-2].equal([[3, 1], [6, 4]])
    assert s[:, -2::-1].equal([[2, 1], [5, 4]])

    s[1, 1] = 22
    assert s[1, 1] == 22
    assert s != s2

    s = FlatCOO.fromobject([[1, None, 3], [None, 5, None]])
    s2 = FlatCOO((2, 3), Buffer.fromobject([0, 2, 4]), Buffer.fromobject([1, 3, 5]))
    assert s == s2
    assert not s.is_full()

    assert s[0, 0] == 1
    assert s[0, 1] == s.unspecified
    assert s[0, 2] == 3
    assert s[1, 0] == s.unspecified
    assert s[1, 1] == 5
    assert s[1, 2] == s.unspecified

    s = FlatCOO.fromobject([[1, 0, 3], [0, 5, 0]], unspecified=0)
    s2 = FlatCOO((2, 3), Buffer.fromobject([0, 2, 4]), Buffer.fromobject([1, 3, 5], unspecified=0))
    assert s == s2
    assert not s.is_full()

    assert s[0, 0] == 1
    assert s[0, 1] == s.unspecified
    assert s[0, 2] == 3
    assert s[1, 0] == s.unspecified
    assert s[1, 1] == 5
    assert s[1, 2] == s.unspecified


def test_CRS():
    s = CRS.fromobject([[1, 2, 3], [4, 5, 6]])
    s2 = CRS((2, 3),
             Buffer.fromobject([0, 3, 6]),
             Buffer.fromobject([0, 1, 2, 0, 1, 2]),
             Buffer.fromobject([1, 2, 3, 4, 5, 6]))
    assert s == s2
    assert s.is_full()
    assert s[0, 0] == 1
    assert s[0, 1] == 2
    assert s[0, 2] == 3
    assert s[1, 0] == 4
    assert s[1, 1] == 5
    assert s[1, 2] == 6
    assert s.shape == (2, 3)

    assert s[0].equal([1, 2, 3])
    assert s[0].size == 3
    assert s[0].shape == (3, )
    assert s[1].equal([4, 5, 6])
    assert s[-1].equal([4, 5, 6])
    assert s[:, 0].equal([1, 4])
    assert s[:, 1].equal([2, 5])
    assert s[:, 2].equal([3, 6])
    assert s[:, -1].equal([3, 6])
    assert s[:, -2].equal([2, 5])

    assert s[0:].equal([[1, 2, 3], [4, 5, 6]])
    assert s[1:].equal([[4, 5, 6]])

    assert s[::-1].equal([[4, 5, 6], [1, 2, 3]])
    assert s[-2::-1].equal([[1, 2, 3]])

    assert s[:, 0:].equal([[1, 2, 3], [4, 5, 6]])
    assert s[:, 1:].equal([[2, 3], [5, 6]])
    assert s[:, 2:].equal([[3], [6]])

    assert s[:, 0:2].equal([[1, 2], [4, 5]])
    assert s[:, :-1].equal([[1, 2], [4, 5]])
    assert s[:, 1:2].equal([[2], [5]])
    assert s[:, ::2].equal([[1, 3], [4, 6]])
    assert s[:, ::-1].equal([[3, 2, 1], [6, 5, 4]])
    assert s[:, ::-2].equal([[3, 1], [6, 4]])
    assert s[:, -2::-1].equal([[2, 1], [5, 4]])

    s[1, 1] = 22
    assert s[1, 1] == 22
    assert s != s2

    s = CRS.fromobject([[1, None, 3], [None, 5, None]])
    s2 = CRS((2, 3),
             Buffer.fromobject([0, 2, 3]),
             Buffer.fromobject([0, 2, 1]),
             Buffer.fromobject([1, 3, 5]))
    assert s == s2
    assert not s.is_full()

    assert s[0, 0] == 1
    assert s[0, 1] == s.unspecified
    assert s[0, 2] == 3
    assert s[1, 0] == s.unspecified
    assert s[1, 1] == 5
    assert s[1, 2] == s.unspecified

    s = CRS.fromobject([[1, 0, 3], [0, 5, 0]], unspecified=0)
    s2 = CRS((2, 3),
             Buffer.fromobject([0, 2, 3]),
             Buffer.fromobject([0, 2, 1]),
             Buffer.fromobject([1, 3, 5], unspecified=0))
    assert s == s2
    assert not s.is_full()

    assert s[0, 0] == 1
    assert s[0, 1] == s.unspecified
    assert s[0, 2] == 3
    assert s[1, 0] == s.unspecified
    assert s[1, 1] == 5
    assert s[1, 2] == s.unspecified


def test_conversion():
    s = Strided.fromobject([[1, 2, 3], [4, 5, 6]])

    coo = s.as_COO()
    fcoo = s.as_FlatCOO()
    crs = s.as_CRS()
    assert coo.as_Strided() == s
    assert coo.as_FlatCOO() == fcoo
    assert coo.as_CRS() == crs
    assert fcoo.as_Strided() == s
    assert fcoo.as_COO() == coo
    assert fcoo.as_CRS() == crs
    assert crs.as_Strided() == s
    assert crs.as_COO() == coo
    assert crs.as_FlatCOO() == fcoo

    s = Strided.fromobject([[1, None, 3], [None, 5, None]])
    coo = s.as_COO()
    fcoo = s.as_FlatCOO()
    crs = s.as_CRS()
    assert coo.as_Strided() == s
    assert coo.as_FlatCOO() == fcoo
    assert coo.as_CRS() == crs
    assert fcoo.as_Strided() == s
    assert fcoo.as_COO() == coo
    assert fcoo.as_CRS() == crs
    assert crs.as_Strided() == s
    assert crs.as_COO() == coo
    assert crs.as_FlatCOO() == fcoo

    s = Strided.fromobject([[1, 0, 3], [0, 5, 0]])
    coo = s.as_COO(unspecified=0)
    fcoo = s.as_FlatCOO(unspecified=0)
    crs = s.as_CRS(unspecified=0)
    assert coo.as_Strided() == s.copy(unspecified=0)
    assert coo.as_FlatCOO() == fcoo
    assert coo.as_CRS() == crs
    assert fcoo.as_Strided() == s.copy(unspecified=0)
    assert fcoo.as_COO() == coo
    assert fcoo.as_CRS() == crs
    assert crs.as_Strided() == s.copy(unspecified=0)
    assert crs.as_COO() == coo
    assert crs.as_FlatCOO() == fcoo


def test_elements():
    s = Strided.fromobject([[1, 2, 3], [4, 5, 6]])
    elements = [((0, 0), 1), ((0, 1), 2), ((0, 2), 3),
                ((1, 0), 4), ((1, 1), 5), ((1, 2), 6)]

    assert list(s.elements()) == elements
    assert list(s.as_COO().elements()) == elements
    assert list(s.as_FlatCOO().elements()) == elements
    assert list(s.as_CRS().elements()) == elements
    assert list(s.as_GCS().elements()) == elements
    assert list(s.as_GCS(storage_class=Strided).elements()) == elements
    assert list(s.as_GCS(storage_class=COO).elements()) == elements
    assert list(s.as_GCS(storage_class=FlatCOO).elements()) == elements
    assert list(s.as_GCS(storage_class=CRS).elements()) == elements
    assert list(s.as_GCS(storage_class=Buffer).elements()) == elements


@pytest.mark.parametrize("storage_clsname", ['Strided', 'COO', 'FlatCOO', 'CRS'])
@pytest.mark.parametrize("reduction", ['2D', '1D', 'T1D'])
def test_2D_GCS(storage_clsname, reduction):
    storage_class = eval(storage_clsname)

    if reduction == '2D':
        rdimensions = ((0,), (1,))
        rstrides = ((1,), (1,))
        roffset = (0, 0)
        partitioning = None
        dimensions = None
        fromobject = storage_class.fromobject
    elif reduction == '1D':
        if storage_class is CRS:
            pytest.skip(f'{reduction}: CRS is for 2D reduction only')
        rdimensions = ((0, 1),)
        rstrides = ((3, 1),)
        roffset = (0,)
        partitioning = ()
        dimensions = (0, 1)

        def fromobject(seq, unspecified=None):
            new_seq = []
            for i in range(len(seq)):
                for j in range(len(seq[0])):
                    new_seq.append(seq[i][j])
            return storage_class.fromobject(new_seq, unspecified=unspecified)

    elif reduction == 'T1D':
        if storage_class is CRS:
            pytest.skip(f'{reduction}: CRS is for 2D reduction only')
        rdimensions = ((1, 0),)
        rstrides = ((2, 1),)
        roffset = (0,)
        partitioning = ()
        dimensions = (1, 0)

        def fromobject(seq, unspecified=None):
            new_seq = []
            for j in range(len(seq[0])):
                for i in range(len(seq)):
                    new_seq.append(seq[i][j])
            return storage_class.fromobject(new_seq, unspecified=unspecified)

    else:
        raise NotImplementedError(reduction)

    seq = [[1, 2, 3], [4, 5, 6]]
    s = GCS.fromobject(seq, dimensions=dimensions, partitioning=partitioning,
                       storage_class=storage_class)
    s2 = GCS((2, 3), rdimensions, rstrides, roffset, fromobject(seq))
    assert s == s2

    assert s[0, 0] == 1
    assert s[0, 1] == 2
    assert s[0, 2] == 3
    assert s[1, 0] == 4
    assert s[1, 1] == 5
    assert s[1, 2] == 6
    assert s.shape == (2, 3)

    s2[1, 1] = 22
    assert s2[1, 1] == 22
    assert s != s2

    seq = [[1, None, 3], [None, 5, None]]
    s = GCS.fromobject(seq, dimensions=dimensions, partitioning=partitioning,
                       storage_class=storage_class)
    s2 = GCS((2, 3), rdimensions, rstrides, roffset, fromobject(seq))

    assert s == s2
    assert s.is_full() == issubclass(storage_class, Strided)

    assert s[0, 0] == 1
    assert s[0, 1] == s.unspecified
    assert s[0, 2] == 3
    assert s[1, 0] == s.unspecified
    assert s[1, 1] == 5
    assert s[1, 2] == s.unspecified

    seq = [[1, 0, 3], [0, 5, 0]]
    s = GCS.fromobject(seq, dimensions=dimensions, partitioning=partitioning,
                       storage_class=storage_class, unspecified=0)
    s2 = GCS((2, 3), rdimensions, rstrides, roffset, fromobject(seq, unspecified=0))
    assert s == s2
    assert s.is_full() == issubclass(storage_class, (Buffer, Strided))

    assert s[0, 0] == 1
    assert s[0, 1] == s.unspecified
    assert s[0, 2] == 3
    assert s[1, 0] == s.unspecified
    assert s[1, 1] == 5
    assert s[1, 2] == s.unspecified


def make_sample(shape, base=0):
    if len(shape) == 1:
        return list([base + i+1 for i in range(shape[0])])
    rows = []
    for i in range(shape[0]):
        rows.append(make_sample(shape[1:], base=base + (i+1) * 10**(len(shape)-1)))
    return rows


def generate_partitioning(N):
    for reduction_dimensionality in range(1, N+1):
        for partitioning in itertools.combinations(tuple(range(1, N)),
                                                   r=reduction_dimensionality-1):
            yield partitioning


def generate_gcs_parameters(N, M):
    if M == 1:
        storage_classes = ('Strided', 'COO', 'FlatCOO', 'Buffer')
    elif M == 2:
        storage_classes = ('Strided', 'COO', 'FlatCOO', 'CRS')
    else:
        storage_classes = ('Strided', 'COO', 'FlatCOO')
    all_permutations = list(itertools.permutations(tuple(range(N))))
    n = len(all_permutations)
    selected_permutations = set()
    selected_permutations.update(all_permutations[:2])
    selected_permutations.update(all_permutations[-2:])
    selected_permutations.update(all_permutations[n//3:n//3+2])
    selected_permutations.update(all_permutations[2*n//3:2*n//3+2])
    selected_permutations.update(all_permutations[n//4:n//4+2])
    selected_permutations.update(all_permutations[n//2:n//2+2])
    selected_permutations.update(all_permutations[3*n//2:3*n//2+2])

    all_partitioning = list(itertools.combinations(tuple(range(1, N)), r=M-1))
    n = len(all_partitioning)
    selected_partitioning = set()
    selected_partitioning.update(all_partitioning[:2])
    selected_partitioning.update(all_partitioning[n//4:n//4+1])
    selected_partitioning.update(all_partitioning[n//2:n//2+1])
    selected_partitioning.update(all_partitioning[3*n//2:3*n//2+1])
    selected_partitioning.update(all_partitioning[-2:])

    for storage_class in storage_classes:
        count = 0
        for dimensions in sorted(selected_permutations):
            count += 1
            if N > 3 and count > 3:
                break
            for partitioning in sorted(selected_partitioning):
                yield (f'{storage_class}-{".".join(map(str,dimensions))}'
                       f'-{".".join(map(str,partitioning))}')


gcs_parameters = []
gcs_parameters.extend(generate_gcs_parameters(2, 1))
gcs_parameters.extend(generate_gcs_parameters(2, 2))
gcs_parameters.extend(generate_gcs_parameters(3, 1))
gcs_parameters.extend(generate_gcs_parameters(3, 2))
gcs_parameters.extend(generate_gcs_parameters(3, 3))
gcs_parameters.extend(generate_gcs_parameters(5, 1))
gcs_parameters.extend(generate_gcs_parameters(5, 2))


@pytest.mark.parametrize("parameters", gcs_parameters)
def test_NtoM_GCS(parameters):
    storage_clsname, dimensions, partitioning = parameters.split('-')
    dimensions = tuple(map(int, dimensions.split('.')))
    partitioning = tuple(map(int, partitioning.split('.'))) if partitioning else ()
    M = len(partitioning) + 1
    N = len(dimensions)
    storage_class = eval(storage_clsname)

    shape = ((2, 3, 4) * N)[:N]

    seq = make_sample(shape)
    sample = Strided.fromobject(seq, shape=shape)

    s = GCS.fromobject(seq, dimensions=dimensions, partitioning=partitioning,
                       storage_class=storage_class)

    assert len(s.shape) == N
    assert len(s.data.shape) == M
    assert sample.equal(s)
    assert s.equal(sample)


@pytest.mark.parametrize("storage_clsname", ['Strided', 'COO', 'FlatCOO', 'CRS', 'GCS'])
def test_slicing(storage_clsname):
    storage_class = eval(storage_clsname)
    s = storage_class.fromobject([[1, 2, 3], [4, 5, 6]])
    s[0] = [11, 12, 13]
    assert s.equal([[11, 12, 13], [4, 5, 6]])
    s[1] = [21, 22, 23]
    assert s.equal([[11, 12, 13], [21, 22, 23]])
    s[0, 1:] = [120, 130]
    assert s.equal([[11, 120, 130], [21, 22, 23]])
    s[:, 1] = [221, 222]
    assert s.equal([[11, 221, 130], [21, 222, 23]])

    s = storage_class.fromobject([[1, 2, 3], [4, 5, 6]])
    assert s[0].equal([1, 2, 3])
    assert s[1].equal([4, 5, 6])
    assert s[:, 0].equal([1, 4])
    assert s[:, 1].equal([2, 5])
    assert s[:, 2].equal([3, 6])
    assert s[1::, 1::].equal([[5, 6]])
    assert s[:1, :2].equal([[1, 2]])
    assert s[0, ::2].equal([1, 3])
    assert s[1, ::2].equal([4, 6])
    assert s[0, ::-1].equal([3, 2, 1])
    assert s[::-1, ::-1].equal([[6, 5, 4], [3, 2, 1]])


def make_slice_samples(size):
    yield 0
    yield size//2
    yield size-1
    yield -1
    yield -size//2
    yield slice(None)
    yield slice(size//2)
    yield slice(None, size//2)
    yield slice(size//4, (3*size)//2)
    yield slice(None, None, 2)
    yield slice(None, None, 3)
    yield slice(size//2, None, 2)
    yield slice(None, size//2, 2)
    yield slice(size//4, (3*size)//2, 2)
    yield slice(size//2, None, 3)
    yield slice(None, size//2, 3)
    yield slice(size//4, (3*size)//2, 3)
    yield slice(None, None, -1)
    yield slice(-2, None, -1)
    yield slice(None, 1, -1)
    yield slice(-2, 1, -1)
    yield slice(None, None, -2)
    yield slice(-2, None, -2)
    yield slice(None, 1, -2)
    yield slice(-2, 1, -2)


gcs_parameters = []
gcs_parameters.extend(generate_gcs_parameters(2, 1))
gcs_parameters.extend(generate_gcs_parameters(2, 2))
gcs_parameters.extend(generate_gcs_parameters(3, 1))
gcs_parameters.extend(generate_gcs_parameters(3, 2))
gcs_parameters.extend(generate_gcs_parameters(5, 1))
gcs_parameters.extend(generate_gcs_parameters(5, 2))


@pytest.mark.parametrize("parameters", gcs_parameters)
def test_GCS_slicing(parameters):
    storage_clsname, dimensions, partitioning = parameters.split('-')
    dimensions = tuple(map(int, dimensions.split('.')))
    partitioning = tuple(map(int, partitioning.split('.'))) if partitioning else ()
    N = len(dimensions)
    storage_class = eval(storage_clsname)
    shape = ((2, 3, 4) * N)[:N]

    seq = make_sample(shape)
    strided = Strided.fromobject(seq)
    s = GCS.fromobject(seq, dimensions=dimensions, partitioning=partitioning,
                       storage_class=storage_class)
    assert strided.equal(s)
    assert s.equal(strided)
    assert strided[:, ::3].equal(s[:, ::3])
    assert strided[::2].equal(s[::2])
    assert strided[::2, ::3].equal(s[::2, ::3])
    assert strided[1::2].equal(s[1::2])
    assert strided[1::2, 2::3].equal(s[1::2, 2::3])
    assert strided[::-2].equal(s[::-2])

    n = len(s.shape)
    all_dimensions = list(range(n))
    selected_dimensions = set()
    selected_dimensions.update(all_dimensions[:1])
    selected_dimensions.update(all_dimensions[n//2:n//2+1])
    selected_dimensions.update(all_dimensions[-1:])

    for k in selected_dimensions:
        for sl1 in make_slice_samples(s.shape[k]):
            sl = (slice(None), ) * k + (sl1,)
            r = s[sl]
            assert isinstance(r, GCS), r
            assert strided[sl].equal(r)
            for k2 in selected_dimensions:
                if k2 <= k:
                    continue
                for sl2 in make_slice_samples(s.shape[k2]):
                    sl = (slice(None), ) * k + (sl1,) + (slice(None),) * (k2 - k - 1) + (sl2,)
                    r = s[sl]
                    expected = strided[sl]
                    if isinstance(r, GCS):
                        assert expected.equal(r)
                    else:
                        assert expected == r, (expected, r, sl)


if __name__ == '__main__':
    import sys
    import pytest
    pytest.main(sys.argv)
