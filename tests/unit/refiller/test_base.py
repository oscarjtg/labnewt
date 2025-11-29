"""Unit tests for Refiller class."""

import numpy as np
import pytest

from labnewt.refiller.base import Refiller


class DummyModel:
    pass


def test_base_refiller_fill_method():
    refiller = Refiller()
    model = DummyModel()
    needs_filling = np.zeros((2, 3), dtype=bool)
    with pytest.raises(NotImplementedError):
        refiller.fill(model, needs_filling)
