import pandas as pd
import numpy as np
import logging


path = 'DataSplit\Train.csv'
df = pd.read_csv(path)


class DataSplit:
    """
    A class to split data into training and testing sets based on year and month ranges.
    """

    @staticmethod
     def log_error(error_message: str, e: Exception) -> None:
        """
        Logs an error message along with the exception details.

        Args:
            error_message (str): Custom error message.
            e (Exception): Exception instance.
        """
        logging.error(f"{error_message}\nException: {e}", exc_info=True)
    @staticmethod
    def split_data(
        df: pd.DataFrame,
        train_year_start: int,
        train_year_end: int,
        train_month_start: int,
        train_month_end: int,
        test_year_start: int,
        test_month_start: int,
        test_month_end: int
    ):
        """
        Splits a DataFrame into training and testing sets based on specified year and month ranges.

        Args:
            df (pd.DataFrame): The dataset to split. It should contain 'year' and 'month' columns.
            train_year_start (int): The starting year for the training set.
            train_year_end (int): The ending year for the training set.
            train_month_start (int): The starting month for the training set.
            train_month_end (int): The ending month for the training set.
            test_year_start (int): The starting year for the testing set.
            test_year_end (int): The ending year for the testing set.
            test_month_start (int): The starting month for the testing set.
            test_month_end (int): The ending month for the testing set.

        Returns:
            Two DataFrame Training Set and Test Set.
        """
        try:
            # Create the training set
            train_set = df[
                # Full years in range
                ((df['year'] > train_year_start) & (df['year'] < train_year_end)) |
                # Start year
                ((df['year'] == train_year_start) & (df['month'] >= train_month_start)) |
                ((df['year'] == train_year_end) &
                (df['month'] <= train_month_end))  # End year
            ]
            # Create the testing set
            test_set = df[
                ((df['year'] >= test_year_start)) &  # Years in range
                (df['month'] >= test_month_start) &
                (df['month'] <= test_month_end)
            ]

            return train_set, test_set
        except Exception as e:
            error_message = (
                "An error occurred while splitting the dataset.\n"
                "Class: DataSplit\n"
                "Method: split_data"
            )
            DataSplit.log_error(error_message, e)
            raise  # Re-raise the exception after logging


# Define training and testing ranges
train_year_start, train_year_end = 2016, 2021
train_month_start, train_month_end = 1, 7
test_year = 2021
test_month_start, test_month_end = 8, 10

# Calling the Function
train_data, test_data = split_data(df, train_year_start, train_year_end, train_month_start,
                                   train_month_end, test_year, test_month_start, test_month_end)
print(train_data.shape)
print(test_data.shape)
