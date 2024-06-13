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
# Modified   : Thursday June 13th 2024 10:56:38 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
"""Numeric Bivariate Analysis Module"""
from typing import Type, Union

import numpy as np
import pandas as pd
from dependency_injector.wiring import Provide, inject

from explorify.container import VisualizeContainer
from explorify.eda.regression.simple import (
    SimpleRegressionAnalysis,
    SimpleRegressionResult,
)
from explorify.eda.stats.inferential.centrality import TTest, TTestResult
from explorify.eda.stats.inferential.correlation import (
    PearsonCorrelationTest,
    PearsonCorrelationTestResult,
    SpearmanCorrelationResult,
    SpearmanCorrelationTest,
)
from explorify.eda.stats.inferential.gof import KSTest, KSTestResult
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
        validate_input(a_name: str, b_name: str) -> None:
            Validates the input variables.
        students_t_test(a_name: str, a_data: np.ndarray, b_name: str, b_data: np.ndarray, varname: str) -> StatTestResult
            Computes the two sample independent student's t-test between two numeric variables.
        correlation_coefficient(a_name: str, b_name: str) -> float:
            Computes the correlation coefficient between two numeric variables.
        regression_analysis(a_name: str, b_name: str) -> dict:
            Performs regression analysis between two numeric variables.
        association_measure(a_name: str, b_name: str) -> float:
            Computes an association measure between two numeric variables.
        visualize(a_name: str, b_name: str) -> None:
            Visualizes the relationship between two numeric variables.
    """

    @inject
    def __init__(
        self,
        data: pd.DataFrame,
        visualizer: Visualizer = Provide[VisualizeContainer.visualizer],
        pearsons_test_cls: Type[PearsonCorrelationTest] = PearsonCorrelationTest,
        spearman_test_cls: Type[SpearmanCorrelationTest] = SpearmanCorrelationTest,
        ks_test_cls: Type[KSTest] = KSTest,
        students_t_test_cls: Type[TTest] = TTest,
        simple_linear_regression_cls: Type[
            SimpleRegressionAnalysis
        ] = SimpleRegressionAnalysis,
    ):
        """
        Initializes the NumericBivariateAnalysis instance with data.

        Args:
            _data (pd.DataFrame): The input data containing the variables.
        """
        self._data = data
        self._visualizer = visualizer
        self._pearsons_test_cls = pearsons_test_cls
        self._spearman_test_cls = spearman_test_cls
        self._ks_test_cls = ks_test_cls
        self._students_t_test_cls = students_t_test_cls
        self._simple_linear_regression_cls = simple_linear_regression_cls

    def validate_input(self, a_name: str, b_name: str) -> None:
        """
        Validates the input variables to ensure they are numeric.

        Args:
            a_name (str): Name of the first numeric variable.
            b_name (str): Name of the second numeric variable.

        Raises:
            ValueError: If either variable is not present in the DataFrame or if they are not numeric.
        """
        if a_name not in self._data.columns or b_name not in self._data.columns:
            raise ValueError("Both variables must be present in the DataFrame.")

        if not pd.api.types.is_numeric_dtype(
            self._data[a_name]
        ) or not pd.api.types.is_numeric_dtype(self._data[b_name]):
            raise ValueError("Both variables must be numeric.")

    def students_t_test(
        self,
        a_name: str,
        a_data: np.ndarray,
        b_name: str,
        b_data: np.ndarray,
        varname: str,
        alpha: float = 0.05,
    ) -> TTestResult:
        """
        Performs a Kolmogorov-Smirnov test.

        The two-sample test compares the underlying distributions of two independent samples.
        This test is valid only for continuous distributions.

        Args:
            a_name (str): Name of a continuous variable in the dataframe
            a_data (np.ndarray): Data belonging to the first group
            b_name (str): Name of a continuous variable in the dataframe
            b_data (np.ndarray): Data belonging to the second group
            varname (str): Name of the variable of interest

        Returns:
            TTestResult: The result of the Student's T-Test.

        Example:
            >>> analysis = BivariateCategoricalNumericAnalysis(data)
            >>> result = analysis.students_t_test_cls(a_name="male",a_data=self.data.loc[self.data["Gender"]=="male"],
                                                      b_name="female", b_data=self.data.loc[self.data["Gender"]=="female"],
                                                      varname="Income")
            >>> print(result)
        """
        self.validate_input(a_name=varname, b_name=varname)
        test = self._students_t_test_cls(
            a_name=a_name,
            a_data=a_data,
            b_name=b_name,
            b_data=b_data,
            varname=varname,
            alpha=alpha,
        )
        test.run()
        return test.result

    def kolmogorov_smirnov_test(self, a_name: str, b_name: str) -> KSTestResult:
        """
        Performs a Kolmogorov-Smirnov test.

        The two-sample test compares the underlying distributions of two independent samples.
        This test is valid only for continuous distributions.

        Args:
            a_name (str): Name of a continuous variable in the dataframe
            b_name (str): Name of a continuous variable in the dataframe


        Returns:
            KSTestResult: The result of the Kolmogorov-Smirnov test.

        Example:
            >>> analysis = BivariateCategoricalNumericAnalysis(data)
            >>> result = analysis.kolmogorov_smirnov_test(a_name=a_name, b_name=b_name)
            >>> print(result)
        """
        self.validate_input(a_name=a_name, b_name=b_name)
        test = self._ks_test_cls(a_name=a_name, b_name=b_name, data=self._data)
        test.run()
        return test.result

    def correlation_coefficient(
        self,
        a_name: str,
        b_name: str,
        normal: bool = True,
    ) -> Union[PearsonCorrelationTestResult, SpearmanCorrelationResult]:
        """
        Computes the correlation coefficient between two numeric variables.

        Args:
            a_name (str): Name of the first numeric variable.
            b_name(str): Name of the second numeric variable.
            normal (bool): True if the variables are assumed to have a normal distribution. False otherwise.

        Returns:
            StatTestResult: The correlation test result
        """
        self.validate_input(a_name=a_name, b_name=b_name)
        if normal:
            analysis = self._pearsons_test_cls(
                a_name=a_name, b_name=b_name, data=self._data
            )
        else:
            analysis = self._spearman_test_cls(
                a_name=a_name, b_name=b_name, data=self._data
            )

        analysis.run()
        return analysis.result

    def regression_analysis(self, a_name: str, b_name: str) -> SimpleRegressionResult:
        """
        Performs regression analysis between two numeric variables.

        Args:
            a_name (str): Name of the independent variable.
            b_name (str): Name of the dependent variable.

        Returns:
            dict: Results of the regression analysis including coefficients, p-values, and R-squared value.
        """
        self.validate_input(a_name=a_name, b_name=b_name)
        analysis = self._simple_linear_regression_cls(
            a_name=a_name, b_name=b_name, data=self._data
        )
        analysis.run()
        return analysis.result

    def visualize(self, a_name: str, b_name: str, title: str = None) -> None:
        """
        Visualizes the relationship between two numeric variables.

        Args:
            a_name (str): Name of the first numeric variable.
            b_name (str): Name of the second numeric variable.
            title (str): Title for the plot. Optional.

        Returns:
            None
        """
        self._visualizer.scatterplot(data=self._data, x=a_name, y=b_name, title=title)
