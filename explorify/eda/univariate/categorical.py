#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Explorify                                                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /categorical.py                                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john@variancexplained.com                                                           #
# URL        : https://github.com/variancexplained/explorify                                       #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday June 8th 2024 11:45:10 am                                                  #
# Modified   : Saturday June 8th 2024 05:04:56 pm                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
import pandas as pd
import numpy as np
# ------------------------------------------------------------------------------------------------ #
class Categorical:
    """Provides methods to perform univariate analysis on categorical data.

    Args:
        data (pd.DataFrame): The DataFrame containing the categorical data.

    Example usage:
        >>> import pandas as pd
        >>> from univariate import Categorical
        >>> data = pd.DataFrame({
        ...     'Category': ['A', 'B', 'A', 'C', 'B', 'A', 'A', 'C']
        ... })
        >>> categorical_analysis = Categorical(data)
        >>> print(categorical_analysis.descriptive_statistics('Category'))
        >>> print(categorical_analysis.frequency_distribution('Category'))
        >>> print(categorical_analysis.diversity_indices('Category'))
        >>> print(categorical_analysis.inequality_measures('Category'))
        >>> print(categorical_analysis.entropy('Category'))
        >>> print(categorical_analysis.spread_measures('Category'))

    """

    def __init__(self, data: pd.DataFrame) -> None:
        self._data = data

    def descriptive_statistics(self, x: str) -> pd.DataFrame:
        """Calculates descriptive statistics for a specific categorical column.

        Args:
            x (str): The column to be analyzed.

        Returns:
            pd.DataFrame: A DataFrame containing the descriptive statistics.
        """
        self._validate_column(x)
        desc_stats = self._data[x].describe()
        return desc_stats.to_frame()

    def frequency_distribution(self, x: str, n: int = None) -> pd.Series:
        """Calculates the frequency distribution for categorical data.

        This yields frequency distribution, including proportions and
        cumulative counts. For large categories, the method takes
        an optional value, 'n', which specifies the top n categories
        to report, based on frequency.

        Args:
            x (str): The column in the dataset containing the categorical
                variable.
            n (int): The number

        Returns:
            pd.DataFrame: A DataFrame containing the frequency distribution.
        """
        self._validate_column(x)
        # Calculate frequency distribution
        freq = self._data[x].value_counts().reset_index()
        freq.columns = [x, "Count"]

        # Calculate cumulative count and proportions
        freq["Cumulative Count"] = freq["Count"].cumsum()
        total_count = freq["Count"].sum()
        freq["Proportion"] = freq["Count"] / total_count
        freq["Cumulative Proportion"] = freq["Proportion"].cumsum()

        if n is None:
            return freq
        else:
            # Top N rows
            top_n = freq.head(n).copy()

            # Row for the rest of the dataset
            if len(freq) > n:
                rest_count = freq.iloc[n:]["Count"].sum()
                rest_cumulative_count = top_n["Cumulative Count"].iloc[-1] + rest_count
                rest_proportion = rest_count / total_count
                rest_cumulative_proportion = (
                    1.0  # because it's the rest, it covers the remaining percentage
                )
                rest_row = pd.DataFrame(
                    {
                        x: [f"Rest of {x}"],
                        "Count": [rest_count],
                        "Cumulative Count": [rest_cumulative_count],
                        "Proportion": [rest_proportion],
                        "Cumulative Proportion": [rest_cumulative_proportion],
                    }
                )
                top_n = pd.concat([top_n, rest_row], ignore_index=True, axis=0)

            # Total row
            total_row = pd.DataFrame(
                {
                    x: ["Total"],
                    "Count": [total_count],
                    "Cumulative Count": [total_count],
                    "Proportion": [1.0],
                    "Cumulative Proportion": [1.0],
                }
            )
            top_n = pd.concat([top_n, total_row], ignore_index=True, axis=0)

            return top_n

    def diversity_indices(self, x: str) -> pd.Series:
        """Calculates diversity indices for a specific categorical column.

        Args:
            x (str): The column to be analyzed.

        Returns:
            pd.Series: A Series containing the diversity indices.
        """
        self._validate_column(x)

        def shannon_entropy(series):
            counts = series.value_counts()
            probs = counts / len(series)
            return -sum(probs * np.log2(probs))

        def simpson_index(series):
            counts = series.value_counts()
            n = len(series)
            return 1 - sum((counts / n) ** 2)

        indices = pd.Series({
            'Shannon Entropy': shannon_entropy(self._data[x]),
            'Simpson Index': simpson_index(self._data[x])
        })
        return indices

    def inequality_measures(self, x: str) -> pd.Series:
        """Calculates inequality measures for a specific categorical column.

        Args:
            x (str): The column to be analyzed.

        Returns:
            pd.Series: A Series containing the inequality measures.
        """
        self._validate_column(x)

        def gini_coefficient(series):
            counts = series.value_counts()
            n = len(series)
            sorted_counts = counts.sort_values()
            cum_counts = sorted_counts.cumsum()
            gini = 1 - (2 * cum_counts - sorted_counts).sum() / (n * counts.sum())
            return gini

        measures = pd.Series({
            'Gini Coefficient': gini_coefficient(self._data[x])
        })
        return measures

    def entropy(self, x: str) -> pd.Series:
        """Calculates entropy for a specific categorical column.

        Args:
            x (str): The column to be analyzed.

        Returns:
            pd.Series: A Series containing the entropy value.
        """
        self._validate_column(x)

        def shannon_entropy(series):
            counts = series.value_counts()
            probs = counts / len(series)
            return -sum(probs * np.log2(probs))

        entropy_value = pd.Series({
            'Entropy': shannon_entropy(self._data[x])
        })
        return entropy_value

    def spread_measures(self, x: str) -> pd.Series:
        """Calculates spread measures for a specific categorical column.

        Args:
            x (str): The column to be analyzed.

        Returns:
            pd.Series: A Series containing the spread measures.
        """
        self._validate_column(x)

        spread_value = pd.Series({
            'Spread': self._data[x].value_counts().max() - self._data[x].value_counts().min()
        })
        return spread_value


    def _validate_column(self, x: str) -> None:
        """Validates the column to ensure it exists and is of type object, category, or string.

        Args:
            x (str): The column to be analyzed.

        Raises:
            ValueError: If the column does not exist or is not of a valid type.
        """
        if x not in self._data.columns:
            raise ValueError(f"Column '{x}' not found in data.")
        if not (isinstance(self._data[x], pd.CategoricalDtype) or
                pd.api.types.is_object_dtype(self._data[x]) or
                pd.api.types.is_string_dtype(self._data[x])):
            raise ValueError(f"Column '{x}' must be of type category, object, or string.")
