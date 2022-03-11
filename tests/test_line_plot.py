import pytest
import matplotlib.pyplot as plt
import numpy as np
from src.line_plot import *


@pytest.mark.mpl_image_compare(baseline_dir='baseline',
                               filename='test_line_plot.png')
def test_line_plot():
    Ks = 10
    mean = np.zeros((Ks-1))
    std = np.zeros((Ks-1))
    x = "x-axis"
    y = "y-axis"
    name = "name of the plot"
    return line_plot(Ks, mean, std, x, y, name)

