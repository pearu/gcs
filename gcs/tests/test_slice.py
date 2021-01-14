
import itertools
from gcs.slice_tools import reduced, nitems, diophantine, solve_index, dot, index_strided


def test_reduced():
    for d in range(1, 10):
        for start in list(range(-d-10, d+10)) + [None]:
            for stop in list(range(-d-10, d+10)) + [None]:
                for step in list(range(-d-10, d+10)) + [None]:
                    if step == 0:
                        continue
                    s = slice(start, stop, step)
                    r = list(range(d))[s]

                    s2 = reduced(s, d)
                    r2 = list(range(d))[s2]

                    assert r == r2, (s, r, s2, r2, d)
                    assert nitems(s, d) == len(r), (s, s2, r, d)

                    r3 = [(s2.start % d + i * s2.step)
                          for i in range(nitems(s2, d))]
                    assert r == r3, (s, r, r3, d)

                    if s2.step > 0:
                        r4 = [i for i in range(d)
                              if (i >= s2.start
                                  and i < s2.stop
                                  and (i - s2.start) % s2.step == 0)]
                    else:
                        r4 = [d - i - 1
                              for i in range(d)
                              if (i >= -s2.start - 1
                                  and i < -s2.stop - 1
                                  and (i + s2.start + 1) % s2.step == 0)]
                    assert r == r4, (r, r4, s2, d)


def test_diophantine():
    for a in [(4, 8), (4, 8, 5), (4, 8, 5, 7),
              (-4, 8), (4, -8, 5), (2, -2, 6, -12, 6, 4), (4, 3*2, 5*3, 4*4, 6*2)]:
        for x in [tuple(range(-3, len(a) - 3)),
                  tuple(range(0, len(a)))]:
            c = sum(a[i] * x[i] for i in range(len(a)))
            x0, xs = diophantine(a, c)
            for t in itertools.product(*[range(-3, 3) for i in range(len(xs))]):
                x1 = tuple(x0[i] + sum(xs[j][i] * t[j] for j in range(len(t)))
                           for i in range(len(a)))
                c1 = sum(x1[i] * a[i] for i in range(len(a)))
                assert c1 == c


def test_solve_index():
    def product(seq):
        return seq[0] * product(seq[1:]) if seq else 1

    assert solve_index(0, (), ()) == ()

    shape = (12, 2)
    strides = (2, 1)
    for index in itertools.product(*map(range, shape)):
        pos = dot(strides, index)
        ind = solve_index(pos, shape, strides)
        assert ind == index

    for shape in [(2, 3), (4, 5), (4, 3, 2), (2, 3, 4, 5), (2,), (5,), (1, 1)]:
        strides = tuple(product(shape[i + 1:]) for i in range(len(shape)))
        for axis in range(len(shape)):
            slices = [slice(None), slice(1), slice(shape[axis]-1),
                      slice(1, None, 2), slice(None, None, -1)]
            for sl in slices:
                sl = reduced(sl, shape[axis])
                slice_shape, slice_strides, slice_offset = index_strided(
                    (None,)*axis + (sl,) + (None,) * (len(shape) - axis - 1), shape, strides)
                for index in itertools.product(*map(range, slice_shape)):
                    pos = dot(slice_strides, index)
                    ind = solve_index(pos, slice_shape, slice_strides)
                    assert ind == index, (ind, index, pos)

            if len(shape) > 1:
                for axis2 in range(axis+1, len(shape)):
                    for sl2 in slices:
                        sl2 = reduced(sl2, shape[axis2])
                        slice_shape, slice_strides, slice_offset = index_strided(
                            (None,)*axis + (sl,) + (None,)*(axis2 - axis - 1) + (sl2,) +
                            (None,) * (len(shape) - axis2 - 1), shape, strides)
                        for index in itertools.product(*map(range, slice_shape)):
                            pos = dot(slice_strides, index)
                            ind = solve_index(pos, slice_shape, slice_strides)
                            assert ind == index, (ind, index, pos)
