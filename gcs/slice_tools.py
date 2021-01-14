"""
Reduction of slice objects
"""
# Author: Pearu Peterson
# Created: June 2020

import itertools


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
            raise ValueError(
                f'index items must be int or slice or None, got {type(ind)} at index {i}')
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
            raise ValueError(
                f'index items must be int or slice or None, got {type(ind)} at index {i}')
    return tuple(new_index)


def extgcd(a, b):
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = extgcd(b % a, a)
    x = y1 - (b//a) * x1
    y = x1
    return gcd, x, y


def _init_diophantine(a):
    """A recurssive helper function for init_diophantine that computes the
    solution of a linear Diophantine equation.
    """
    if len(a) <= 1:
        return []
    d, x0, y0 = extgcd(a[0], a[1])
    xs, ys = -(a[1] // d), a[0] // d
    next_data = _init_diophantine((d,) + a[2:])
    if next_data:
        zs = next_data[-1][-1]
        z0 = next_data[-1][-2]
    else:
        zs = ((),)
        z0 = (1,)
    s0 = (xs,) + tuple(x0*z for z in zs[0])
    s1 = (ys,) + tuple(y0*z for z in zs[0])
    sr = tuple((0,) + z for z in zs[1:])
    ds = (s0, s1) + sr
    z0 = (x0 * z0[0], y0 * z0[0]) + z0[1:]
    return next_data + [(d, x0, y0, xs, ys, z0, ds)]


def init_diophantine(a, shape=None):
    """Initialize data for solving the linear Diophantine equation in the form

      sum(a[i] * x[i], i=0..) = c

    Return (d, z0, zs, D, b, s) such that for any c, the solution can
    be found as follows:

    If c does not divide by d (c % d != 0), no solution exists.

    Otherwise, the solution is in the form

      x[i] = z0[i] * c // d + t[j] * zs[j][i], j=0..len(t)-1

    where t is a sequence of arbitrary integers: len(t) == len(a) - 1.

    The parameters D, b, s allow restricting the solution to a certain
    hypercube, see solve_index for more information.

    """
    if len(a) == 0:
        return 1, (), (), (), (), None

    if len(a) == 1:
        d = a[0]
        return d, (1,), (), (), (), shape

    data = _init_diophantine(a)

    d, z0, zs = data[0][0], data[-1][-2], data[-1][-1]
    zs = tuple(zip(*zs))  # transpose

    # See solve_index for more info about A
    A = dict()
    N = len(a)
    n = len(zs)
    m = 2*N + 1 if shape is None else 2*N + 2

    for i in range(N):
        for j in range(n):
            A[i, j] = zs[j][i]
            A[N + i, j] = -zs[j][i]
        for j in range(2 * N - n):
            A[i, n + j] = int(i == j)
            A[i + N, n + j] = int(i == j - N)
        A[i, 2*N] = -z0[i]
        A[N + i, 2*N] = z0[i]
        if shape is not None:
            A[i, 2*N+1] = 0
            A[N + i, 2*N+1] = -shape[i] + 1

    # Apply integral Gauss elimination
    for i in range(2*N):
        aii = A[i, i]
        assert aii
        for k in range(2*N):
            if i == k:
                continue
            aki = A[k, i]
            if aki != 0:
                for j in range(m):
                    A[k, j] = A[i, j] * aki - A[k, j] * aii
            assert A[k, i] == 0

    # Ensure positive diagonal and row-wise unit gcd
    for i in range(n):
        g = A[i, i]
        for j in range(2*N, m):
            g, _, _ = extgcd(g, A[i, j])
        if g != 1:
            A[i, i] = A[i, i] // g
            for j in range(2*N, m):
                A[i, j] = A[i, j] // g

    D = tuple(A[i, i] for i in range(n))
    b = tuple(A[i, 2*N] for i in range(n))
    s = tuple(A[i, 2*N + 1] for i in range(n)) if shape is not None else None
    return d, z0, zs, D, b, s


def diophantine(a, c):
    """
    Return the fundamental solution to the linear Diophantine equation
      sum(a[i] * x[i], i=0..) = c
    as
      x[i] = x0[i] + t[j] * zs[i][j], j=0..len(t)-1
    where t is a sequence of arbitrary integers: len(t) == len(a) - 1.
    """
    d, z0, zs, D, b, s = init_diophantine(a)
    if c % d:
        return None, ()
    k = c // d
    return tuple(z * k for z in z0), zs


def dot(a, b):
    return sum(a_ * b_ for a_, b_ in zip(a, b))


def axpy(a, x, y):
    return tuple(a * x_ + y_ for x_, y_ in zip(x, y))


def show(A):
    if A:
        n, m = map(max, zip(*A))
        n += 1
        m += 1
        w = max(map(len, map(str, A.values())))
    else:
        n, m = 0, 0
        w = 0
    fmt = '%'+str(w+1)+'s'
    print('[')
    for i in range(n):
        row = []
        for j in range(m):
            row.append(fmt % (A.get((i, j), '-')))
        print(''.join(row))
    print(']')


def solve_index(pos, shape, strides, data=None):
    """Solve linear Diophantine equation

      sum(index[i] * strides[i] for i in range(len(shape))) == pos

    for 0 <= index[i] < shape[i].

    The solution can be represented as

      index = z0 * k + sum(t[j] * zs[j])

    where k = pos // d; d, z0, zs are parameters that depend only on
    strides. Here t is an (N-1)-sequence of integers that, in general,
    are arbitrary but restricted when requiring 0 <= index < shape to
    hold.

    solve_index aims at finding an unique solution to the above
    equation which boils down to finding the set of integers in t
    with the shape constraints holding:

      sum(t[j] *  zs[j]) >= -z0 * k
      sum(t[j] * -zs[j]) >=  z0 * k - shape + 1

    This system can be written as

      A t >= b            2*N inequalities for N-1 variables t

    or

      A t + I s = b       2*N equalities with N+1 slack variables s
    """
    if data is None:
        data = init_diophantine(strides, shape=shape)
    d, z0, zs, D, b, s = data
    assert not shape or s is not None
    if pos % d:
        return  # no solution exists

    k = pos // d
    n = len(zs)
    # While t is close to a solution, some items may be off by
    # 1. Hence the dt loop.
    for dt in itertools.product(*(((0, 1),)*n)):
        index = axpy(k, z0, [0]*len(z0))
        for i in range(n):
            if s is not None:
                t = (b[i] * k + s[i])//D[i] + dt[i]
            else:
                t = (b[i] * k)//D[i] + dt[i]
            index = axpy(t, zs[i], index)
        for i in range(len(shape)):
            if not (0 <= index[i] < shape[i]):
                break
        else:
            return index
    return


get_solve_index_data = init_diophantine
