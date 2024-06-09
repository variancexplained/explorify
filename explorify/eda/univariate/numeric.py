#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Explorify                                                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /numeric.py                                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john@variancexplained.com                                                           #
# URL        : https://github.com/variancexplained/explorify                                       #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday June 8th 2024 04:29:10 pm                                                  #
# Modified   : Saturday June 8th 2024 05:04:12 pm                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
"""Univariate Numeric Module"""


import pandas as pd
import numpy as np
import scipy.stats as stats
# ------------------------------------------------------------------------------------------------ #
class Numeric:
    """
    A class for performing various univariate analyses on numeric data.

    Attributes:
    -----------
    data : pd.DataFrame
        The DataFrame containing the numeric data for analysis.

    Methods:
    --------
    descriptive_statistics(x: str) -> pd.DataFrame
        Calculate descriptive statistics including mean, median, mode, range,
        standard deviation, skewness, kurtosis, and quantiles for a specified column.

    variance(x: str) -> float
        Calculate the variance of the specified column.

    iqr(x: str) -> float
        Calculate the interquartile range of the specified column.

    mad(x: str) -> float
        Calculate the mean absolute deviation of the specified column.

    coefficient_of_variation(x: str) -> float
        Calculate the coefficient of variation of the specified column.

    standard_error(x: str) -> float
        Calculate the standard error of the mean of the specified column.

    Example Usage:
    --------------
    >>> data = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
    >>> nu = NumericUnivariate(data)
    >>> nu.descriptive_statistics('A')
       count  mean  std  min  25%  50%  75%  max  range  skewness  kurtosis
    A    5.0   3.0  1.581139  1.0  2.0  3.0  4.0  5.0    4.0       0.0       -1.3
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the NumericUnivariate object with a DataFrame.

        Args:
        -----
        data : pd.DataFrame
            The DataFrame containing the numeric data for analysis.
        """
        self._data = data

    def descriptive_statistics(self, x: str) -> pd.DataFrame:
        """
        Calculate descriptive statistics including mean, median, mode, range,
        standard deviation, skewness, kurtosis, and quantiles for a specified column.

        Args:
        -----
        x : str
            The column name of the numeric data to analyze.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the descriptive statistics.
        """
        self._validate_column(x)
        series = self._data[x]
        df = series.describe(percentiles=[.25, .5, .75]).to_frame().T
        df["variance"] = self._data[x].var()
        df['range'] = df['max'] - df['min']
        df['skewness'] = stats.skew(series)
        df['kurtosis'] = stats.kurtosis(series)

        # Mode handling
        mode_result = stats.mode(series)
        df['mode'] = mode_result.mode if mode_result.count > 0 else np.nan

        return df.T

    def iqr(self, x: str) -> float:
        """
        Calculate the interquartile range of the specified column.

        Args:
        -----
        x : str
            The column name of the numeric data to analyze.

        Returns:
        --------
        float
            The interquartile range of the data.
        """
        self._validate_column(x)
        series = self._data[x]
        return stats.iqr(series)

    def mad(self, x: str) -> float:
        """
        Calculate the mean absolute deviation of the specified column.

        Args:
        -----
        x : str
            The column name of the numeric data to analyze.

        Returns:
        --------
        float
            The mean absolute deviation of the data.
        """
        self._validate_column(x)
        series = self._data[x]
        return np.mean(np.abs(series - np.mean(series)))

    def coefficient_of_variation(self, x: str) -> float:
        """
        Calculate the coefficient of variation of the specified column.

        Args:
        -----
        x : str
            The column name of the numeric data to analyze.

        Returns:
        --------
        float
            The coefficient of variation of the data, expressed as a percentage.
        """
        self._validate_column(x)
        series = self._data[x]
        mean = np.mean(series)
        std_dev = np.std(series, ddof=1)
        return (std_dev / mean) * 100

    def standard_error(self, x: str) -> float:
        """
        Calculate the standard error of the mean of the specified column.

        Args:
        -----
        x : str
            The column name of the numeric data to analyze.

        Returns:
        --------
        float
            The standard error of the mean of the data.
        """
        self._validate_column(x)
        series = self._data[x]
        std_dev = np.std(series, ddof=1)
        return std_dev / np.sqrt(len(series))

    def _validate_column(self, x: str) -> None:
        """
        Validate that the specified column exists in the DataFrame and is numeric.

        Args:
        -----
        x : str
            The column name to validate.

        Raises:
        -------
        ValueError:
            If the column does not exist in the DataFrame or is not numeric.
        """
        if x not in self._data.columns:
            raise ValueError(f"Column '{x}' does not exist in the DataFrame.")
        if not pd.api.types.is_numeric_dtype(self._data[x]):
            raise ValueError(f"Column '{x}' is not numeric.")