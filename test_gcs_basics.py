
import random
import itertools

from gcs_basics import *

def test_reduction_1d():
    for shape in [(1,), (5,), (3, 4), (3, 4, 5, 2, 1)]:
        N = len(shape)
        for dimensions in itertools.permutations(range(N)):
            metadata = create_reduction_metadata(shape, dimensions, ())
            for indices in itertools.product(*[range(d) for d in shape]):
                reduction_indices = reduction(metadata, indices)
                assert len(reduction_indices) == 1
                assert promotion(metadata, reduction_indices) == indices

def test_reduction_2d():
    for shape in [(3, 4), (3, 4, 2), (3, 4, 5, 2, 1)]:
        N = len(shape)
        for dimensions in itertools.permutations(range(N)):
            for p in range(1, N):
                metadata = create_reduction_metadata(shape, dimensions, (p,))
                for indices in itertools.product(*[range(d) for d in shape]):
                    reduction_indices = reduction(metadata, indices)
                    assert len(reduction_indices) == 2
                    assert promotion(metadata, reduction_indices) == indices

def test_reduction_3d():
    for shape in [(3, 4, 2), (3, 4, 5, 2, 1)]:
        N = len(shape)
        for dimensions in itertools.permutations(range(N)):
            for p1 in range(1, N-1):
                for p2 in range(p1+1, N):
                    metadata = create_reduction_metadata(shape, dimensions, (p1, p2))
                    for indices in itertools.product(*[range(d) for d in shape]):
                        reduction_indices = reduction(metadata, indices)
                        assert len(reduction_indices) == 3
                        assert promotion(metadata, reduction_indices) == indices

def test_compress_indices():

    assert compress_indices([0, 1, 2, 3, 4], 5) == [0, 1, 2, 3, 4, 5]
    assert compress_indices([0, 1, 2], 3) == [0, 1, 2, 3]
    assert compress_indices([0, 1, 1], 3) == [0, 1, 3, 3]
    assert compress_indices([0, 0, 2], 3) == [0, 2, 2, 3]
    assert compress_indices([1, 1, 2], 3) == [0, 0, 2, 3]
    assert compress_indices([0, 0, 1, 1, 1, 2, 2, 2, 2], 3) == [0, 2, 5, 9]
    assert compress_indices([0, 0, 2, 2, 2, 2], 3) == [0, 2, 2, 6]

    assert compress_indices([], 3) == [0, 0, 0, 0]
    assert compress_indices([0], 3) == [0, 1, 1, 1]
    assert compress_indices([2], 3) == [0, 0, 0, 1]
    
def test_example_5d():

    dimensions = (2, 4, 1, 0, 3)
    shape = (2, 3, 4, 2, 3)

    metadata = create_reduction_metadata(shape, dimensions, (2,))
    
    all_indices = list(itertools.product(*[range(d) for d in shape]))
    sample_indices = sorted(random.sample(all_indices, k=9))
    print(metadata)
    d = {}
    for i, indices in enumerate(sample_indices):
        reduction_indices = reduction(metadata, indices)
        print(indices, i+1, reduction_indices)
        d[reduction_indices] = i+1

    rows, cols = metadata['reduction_shape']
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(str(d.get((i, j), 'X')))
        print(' '.join(row))

    row_indices = []
    col_indices = []
    values = []
    for i, j in sorted(d):
        row_indices.append(i)
        col_indices.append(j)
        values.append(d[(i, j)])

    print(' '.join(map(str, row_indices)))
    print(' '.join(map(str, col_indices)))
    print(' '.join(map(str, values)))
    


def test_example_3d():

    shape = (2, 3, 4)

    for dimensions, partitioning in [
            ((0, 1, 2), (2, )),
            ((0, 1, 2), (1, )),
            ((2, 1, 0), (1, )),
    ]:
        print('-'*10)
        print(f'{dimensions=}, {partitioning=}')

        metadata = create_reduction_metadata(shape, dimensions, partitioning)

        # all_indices = list(itertools.product(*[range(d) for d in shape]))
        # sample_indices = sorted(random.sample(all_indices, k=9))

        sample_indices_values = [
            ((0, 0, 1), 1),
            ((0, 0, 2), 2),
            ((0, 0, 3), 3),
            ((0, 2, 1), 4),
            ((1, 0, 0), 5),
            ((1, 0, 3), 6),
            ((1, 2, 0), 7),
            ((1, 2, 2), 8),
            ((1, 2, 3), 9),
        ]

        print(metadata)
        d = {}

        for indices, value in sample_indices_values:
            reduction_indices = reduction(metadata, indices)
            print(indices, '|', value)
            d[reduction_indices] = value

        for indices, value in sample_indices_values:
            row_index, col_index = reduction(metadata, indices)
            print(indices, '|', row_index, '             |', col_index)



        rows, cols = metadata['reduction_shape']
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(str(d.get((i, j), 'X')))
            print(' '.join(row))

        row_indices = []
        col_indices = []
        values = []
        for i, j in sorted(d):
            row_indices.append(i)
            col_indices.append(j)
            values.append(d[(i, j)])

        print(' '.join(map(str, row_indices)))
        print(' '.join(map(str, col_indices)))
        print(' '.join(map(str, values)))


def test_example_full_3d():

    shape = (2, 3, 4)
    N = len(shape)

    print()
    print(f'| Row / Column dimensions   | 2-D reduction array {"":31} |')            
    for p in range(1, N):
        for dimensions in itertools.permutations(range(N)):

            partitioning = (p,)

            metadata = create_reduction_metadata(shape, dimensions, partitioning)

            row_dimensions, col_dimensions = metadata['reduction_dimensions']

            row_dims = '`[' + ', '.join(map(str, row_dimensions)) + ']`'
            col_dims = '`[' + ', '.join(map(str, col_dimensions)) + ']`'

            print(f'| {"-":->3} | {"-":->3} | ')
            print(f'| {row_dims:10} / {col_dims:10}   | ', end='')
            tab = f'| {"":>25} | '
            sample_indices_values = []
            for indices in itertools.product(*[range(d) for d in shape]):
                value = indices[0] * 100 + indices[1] * 10 + indices[2]
                sample_indices_values.append((indices, value))

            d = {}

            for indices, value in sample_indices_values:
                reduction_indices = reduction(metadata, indices)
                d[reduction_indices] = value

            rows, cols = metadata['reduction_shape']
            for i in range(rows):
                row = []
                for j in range(cols):
                    value = d.get((i, j), 'X')
                    svalue = f'{value:03}'
                    row.append(svalue)
                rowline = '`' + ' '.join(row) + '`'
                print(tab if i else '', f'{rowline:50} |')
            continue
            row_indices = []
            col_indices = []
            values = []
            for i, j in sorted(d):
                row_indices.append(i)
                col_indices.append(j)
                values.append(d[(i, j)])

            print(' '.join(map(str, row_indices)))
            print(' '.join(map(str, col_indices)))
            print(' '.join(map(str, values)))
