#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Explorify                                                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /explorify/eda/bivariate/numeric.py                                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john@variancexplained.com                                                           #
# URL        : https://github.com/variancexplained/explorify                                       #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday June 9th 2024 03:08:05 pm                                                    #
# Modified   : Sunday June 9th 2024 04:19:00 pm                                                    #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
"""Numeric Bivariate Analysis Module"""
import pandas as pd
from dependency_injector.wiring import Provide, inject
from scipy import stats

from explorify.container import VisualizeContainer
from explorify.eda.stats.inferential.base import StatTestResult
from explorify.eda.stats.inferential.correlation import PearsonCorrelationTest
from explorify.eda.visualize.visualizer import Visualizer


# ------------------------------------------------------------------------------------------------ #
#                              NUMERIC BIVARIATE ANALYSIS                                          #
# ------------------------------------------------------------------------------------------------ #
class NumericBivariateAnalysis:
    """
    Base class for analyzing bivariate relationships between numeric variables.

    Args:
        _data (pd.DataFrame): The input data containing the variables.

    Attributes:
        _data (pd.DataFrame): The input data containing the variables.

    Methods:
        validate_input(var1: str, var2: str) -> None:
            Validates the input variables.
        correlation_coefficient(var1: str, var2: str) -> float:
            Computes the correlation coefficient between two numeric variables.
        regression_analysis(var1: str, var2: str) -> dict:
            Performs regression analysis between two numeric variables.
        association_measure(var1: str, var2: str) -> float:
            Computes an association measure between two numeric variables.
        visualize(var1: str, var2: str) -> None:
            Visualizes the relationship between two numeric variables.
    """

    @inject
    def __init__(
        self,
        data: pd.DataFrame,
        visualizer: Visualizer = Provide[VisualizeContainer.visualizer],
        pearsons_test_cls: type[PearsonCorrelationTest] = PearsonCorrelationTest,
    ):
        """
        Initializes the NumericBivariateAnalysis instance with data.

        Args:
            _data (pd.DataFrame): The input data containing the variables.
        """
        self._data = data
        self._visualizer = visualizer
        self._pearsons_test_cls = pearsons_test_cls

    def validate_input(self, var1: str, var2: str) -> None:
        """
        Validates the input variables to ensure they are numeric.

        Args:
            var1 (str): Name of the first numeric variable.
            var2 (str): Name of the second numeric variable.

        Raises:
            ValueError: If either variable is not present in the DataFrame or if they are not numeric.
        """
        if var1 not in self._data.columns or var2 not in self._data.columns:
            raise ValueError("Both variables must be present in the DataFrame.")

        if not pd.api.types.is_numeric_dtype(
            self._data[var1]
        ) or not pd.api.types.is_numeric_dtype(self._data[var2]):
            raise ValueError("Both variables must be numeric.")

    def correlation_coefficient(self, var1: str, var2: str) -> StatTestResult:
        """
        Computes the correlation coefficient between two numeric variables.

        Args:
            var1 (str): Name of the first numeric variable.
            var2 (str): Name of the second numeric variable.

        Returns:
            StatTestResult: The correlation test result
        """
        self.validate_input(var1=var1, var2=var2)
        analysis = self._pearsons_test_cls(data=self._data, a=var1, b=var2)
        analysis.run()
        return analysis.result

    def regression_analysis(self, var1: str, var2: str) -> dict:
        """
        Performs regression analysis between two numeric variables.

        Args:
            var1 (str): Name of the independent variable.
            var2 (str): Name of the dependent variable.

        Returns:
            dict: Results of the regression analysis including coefficients, p-values, and R-squared value.
        """
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            self._data[var1], self._data[var2]
        )
        return {
            "slope": slope,
            "intercept": intercept,
            "r_value": r_value,
            "p_value": p_value,
            "std_err": std_err,
        }

    def visualize(self, var1: str, var2: str, title: str = None) -> None:
        """
        Visualizes the relationship between two numeric variables.

        Args:
            var1 (str): Name of the first numeric variable.
            var2 (str): Name of the second numeric variable.
            title (str): Title for the plot. Optional.

        Returns:
            None
        """
        self._visualizer.scatterplot(data=self._data, x=var1, y=var2, title=title)
