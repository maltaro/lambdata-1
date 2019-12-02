"""lambdata - a collection of data science helper functions"""

import pandas as pd
import numpy as np

# sample code

# sample datasets
ONES = pd.DataFrame(np.ones(10))
ZEROS = pd.DataFrame(np.zeros(10))


# sample functions
def increment(x):
    x = x + 1
