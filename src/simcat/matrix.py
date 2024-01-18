import pandas as pd
import numpy as np
from .utils import load_matrix


class Matrix:
    """Semantic memory of Agent
    Args:
        filename (str): file where similarity matrix is stored
        path (str): relative path to folder where matrix is located
        name (str, optional): optional name for model (if None, uses filename)
    """

    def __init__(self, filename, path=None, name=None):
        model = load_matrix(filename=filename, path=path)
        self.data = model.copy()
        self.name = name or filename.split(".")[0]

    @property
    def words(self):
        return list(self.data.index)
