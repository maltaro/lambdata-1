"""
utility functions for working with DataFrames
"""

import numpy as np
import pandas as pd

TEST_DF = pd.DataFrame([1, 2, 3, 4, 5, 6])
TEST_DF_NANS = pd.DataFrame([1, np.nan, 3, 4, np.nan, 6])


def print_nulls(df):
    """
    Check a dataframe for nulls, print them in a "pretty" format
    """

    # Find the number of nulls in each column
    nulls = df.isnull().sum()

    # Make a dataframe from the null series
    nulls_data = list(zip(nulls.index, nulls))
    nulls_columns = ['Column', 'Number of Missing Values']
    nulls_df = pd.DataFrame(nulls_data, columns=nulls_columns)

    # Set the index of the dataframe to the column name
    nulls_df = nulls_df.set_index('Column')

    return nulls_df


def add_list_to_df(list_to_add, df):
    """
    Take a list, turn into a Series, add it to a DataFrame as a new column
    """

    df = df.copy()

    # Convert list to Series
    new_series = pd.Series(list_to_add)

    # Add new Series to the dataframe
    df = pd.concat([df, new_series], axis=1, ignore_index=True)

    return df
