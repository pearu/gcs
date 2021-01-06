from gcs.slice_tools import reduced, nitems

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
