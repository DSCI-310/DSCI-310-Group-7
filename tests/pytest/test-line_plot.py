import pytest
import numpy as np
from src.line_plot import *

@pytest.mark.mpl_image_compare
def test_line_plot():
    k = 10
    mean = array[1,2,3]
    std = np.zeros((Ks-1))
    x = "x-axis"
    y = "y-axis"
    name = "name of the plot"
    return lp.line_plot(k, mean, std, x, y, name)
