import numpy as np

CASES = ("Example A", "Example B")
ORIGINS = (.55, .65, .75, .90)
PRIMARY = .65
TEST = 1.45
CHECK_ORIGIN = ORIGINS[-1]
TARGET = 2.0
EDGES = np.linspace(0., 3.2, 11)
SLOPE_SD = np.linspace(.36, .08, len(EDGES) - 1)
FOCAL = np.array([PRIMARY, 1.0, TEST, TARGET, EDGES[-1]])
