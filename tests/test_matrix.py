from simcat import Matrix
import random
import math


def test_matrix():
    matrix = Matrix("data/matrix.tsv")
    assert matrix.data.shape == (240, 240)
    assert len(matrix.words) == 240
    r_idx = random.randint(0, 239)
    assert math.isnan(matrix.data.values[r_idx, r_idx])
