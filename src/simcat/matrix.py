import pandas as pd
import numpy as np
from .utils import load_matrix


class Matrix:
    """Semantic memory of Agent
    Args:
        filename (str): file where similarity matrix is stored
    """

    def __init__(self, filename):
        model = load_matrix(filename=filename)
        self.data = model.copy()

    @property
    def words(self):
        return list(self.data.index)
