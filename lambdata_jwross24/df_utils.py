import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels


class DataFrameUtils:
    """
    utility functions for working with DataFrames
    """

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

    def plot_confusion_matrix(y_true, y_pred):
        """
        Plot a confusion matrix with labels for each row and column
        """

        # Generate the labels for the data
        labels = unique_labels(y_true)

        # Generate columns and row names from the labels
        columns = [f'Predicted {label}' for label in labels]
        index = [f'Actual {label}' for label in labels]

        # Plot the confusion matrix with the labels
        cm = confusion_matrix(y_true, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalize
        table = pd.DataFrame(cm, columns=columns, index=index)
        return sns.heatmap(table, annot=True, fmt='.2f', cmap='magma')

    def train_val_test(df, generate_val=True, test_ratio=0.2, val_ratio=0.2,
                       random_state=42):
        """
        Generate train/(val)/test splits from a DataFrame
        """

        val = None
        train, test = train_test_split(df, test_size=test_ratio,
                                       random_state=random_state)

        # Do we want a validation set?
        if generate_val:
            train, val = train_test_split(train, test_size=val_ratio,
                                          random_state=random_state)

        return train, val, test
