#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Explorify                                                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /explorify/eda/stats/inferential/association.py                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/variancexplained/explorify                                       #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 29th 2023 11:52:02 pm                                              #
# Modified   : Sunday June 9th 2024 04:27:48 pm                                                    #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from dependency_injector.wiring import Provide, inject
from scipy import stats

from explorify.container import VisualizeContainer
from explorify.eda.stats.inferential.base import StatAnalysis, StatTestResult
from explorify.eda.visualize.visualizer import Visualizer


# ================================================================================================ #
#                                       KENDALL'S TAU                                              #
# ================================================================================================ #
# ------------------------------------------------------------------------------------------------ #
#                                 KENDALL'S TAU MEASURE OF CORRELATION                             #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class KendallsTau(StatTestResult):
    name: str = f"Kendall's \u03c4"  # noqa
    data: pd.DataFrame = None
    a: str = None
    b: str = None
    n: int = None
    strength: str = None
    pvalue: float = None
    visualizer: Visualizer = None

    @inject
    def __post_init__(
        self, visualizer: Visualizer = Provide[VisualizeContainer.visualizer]
    ) -> None:
        self.visualizer = visualizer

    def plot(self) -> None:  # pragma: no cover
        self.visualizer.kendallstau(
            data=self._data,
            a=self.a,
            b=self.b,
            value=self.value,
            thresholds=self.thresholds,
            interpretation=self.interpretation,
        )

    def report(self) -> str:
        return f"Kendall's Tau Test of Association between {self.a.capitalize()} and {self.b.capitalize()} \u03C4={round(self.value,2)},{self._report_pvalue(self.pvalue)}."


# ------------------------------------------------------------------------------------------------ #
#                                   CRAMERS V ANALYSIS                                             #
# ------------------------------------------------------------------------------------------------ #
class KendallsTauAnalysis(StatAnalysis):
    """Kendall's Tau Measures the degree of correlation between two ordinal variables.

    Args:
        _data (pd.DataFrame): The DataFrame containing the variables of interest.
        a (str): The name of an ordinal variable in data.
        b (str): The name of an ordinal variable in data.

    """

    __id = "kendallstau"

    @inject
    def __init__(
        self,
        data: pd.DataFrame,
        a: str = None,
        b: str = None,
        variant: str = "c",
        alternative: str = "two-sided",
        ordinal_a: bool = False,
        ordinal_b: bool = False,
    ) -> None:
        super().__init__()
        self._data = data
        self._a = a
        self._b = b
        self._variant = variant
        self._alternative = alternative
        self._ordinal_a = ordinal_a
        self._ordinal_b = ordinal_b
        self._thresholds = np.array([-1, -0.5, -0.3, 0.0, 0.3, 0.5, 1.0])
        self._labels = [
            "Strong",
            "Moderate",
            "Weak",
            "Weak",
            "Moderate",
            "Strong",
        ]

    @property
    def result(self) -> KendallsTau:
        """Returns the Cramer's V Measure object."""
        return self._result

    def run(self) -> None:
        """Performs the statistical test and creates a result object."""
        data = self._data.copy()

        if self._ordinal_a:
            data[self._a] = data[self._a].sort_values()
        if self._ordinal_b:
            data[self._b] = data[self._b].sort_values()

        a = data[self._a].values
        b = data[self._b].values

        statistic, pvalue = stats.kendalltau(
            x=a, y=b, variant=self._variant, alternative=self._alternative
        )

        strength = self._labels[np.argmax(np.where(self._thresholds < statistic))]

        # Create the result object.
        self._result = KendallsTau(
            data=data,
            a=self._a,
            b=self._b,
            n=len(a),
            value=statistic,
            strength=strength,
            pvalue=pvalue,
        )


# ================================================================================================ #
#                                         CRAMER'S V                                               #
# ================================================================================================ #


# ------------------------------------------------------------------------------------------------ #
#                                 CRAMERS V MEASURE OF ASSOCIATION                                 #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class CramersV(StatTestResult):
    name: str = "Cramer's V"
    strength: str = None
    data: pd.DataFrame = None
    a: str = None
    b: str = None
    n: int = None
    dof: int = None
    x2alpha: float = None
    x2: float = None
    x2dof: float = None
    pvalue: float = None
    expected_freq: np.array = None
    x2result: str = None
    visualizer: Visualizer = None

    @inject
    def __post_init__(
        self, visualizer: Visualizer = Provide[VisualizeContainer.visualizer]
    ) -> None:
        self.visualizer = visualizer

    def report(self) -> str:
        return f"(X\u00b2 ({self.dof}, n={self.data.shape[0]})={round(self.value,2)}, {self._report_pvalue(self.pvalue)}, phi={round(self.value,2)}."


# ------------------------------------------------------------------------------------------------ #
#                                   CRAMERS V ANALYSIS                                             #
# ------------------------------------------------------------------------------------------------ #
class CramersVAnalysis(StatAnalysis):
    """Cramer's V Analysis of the Association between two Nominal Variables.

    The CramersVAnalysis class provides the association between two nominal variables.

    Args:
        _data (pd.DataFrame): The DataFrame containing the variables of interest.
        x (str): The name of the independent variable in data.
        y (str): The name of the dependent variable in data.
        alpha (float): The level of significance for the independence hypothesis test. Default = 0.05
    """

    __id = "cramersv"

    @inject
    def __init__(
        self,
        data: pd.DataFrame,
        a: str = None,
        b: str = None,
        alpha: float = 0.05,
        ordinal_a: bool = False,
        ordinal_b: bool = False,
    ) -> None:
        super().__init__()
        self._data = data
        self._a = a
        self._b = b
        self._alpha = alpha
        self._ordinal_a = ordinal_a
        self._ordinal_b = ordinal_b
        self._thresholds = {
            1: [0.0, 0.1, 0.3, 0.5, 1.0],
            2: [0.0, 0.07, 0.21, 0.35, 1.0],
            3: [0.0, 0.06, 0.17, 0.29, 1.0],
            4: [0.0, 0.05, 0.15, 0.25, 1.0],
            5: [0.0, 0.04, 0.13, 0.22, 1.0],
            6: [0.0, 0.04, 0.13, 0.22, 1.0],
            7: [0.0, 0.04, 0.13, 0.22, 1.0],
            8: [0.0, 0.04, 0.13, 0.22, 1.0],
            9: [0.0, 0.04, 0.13, 0.22, 1.0],
            10: [0.0, 0.04, 0.13, 0.22, 1.0],
        }
        self._labels = ["Negligible", "Small", "Moderate", "Large"]

    @property
    def result(self) -> CramersV:
        """Returns the Cramer's V Result object."""
        return self._result

    def run(self) -> None:
        """Performs the statistical test and creates a result object."""

        data = self._data.copy()

        if self._ordinal_a:
            data[self._a] = data[self._a].sort_values()
        if self._ordinal_b:
            data[self._b] = data[self._b].sort_values()

        crosstab = pd.crosstab(data[self._a], data[self._b])

        statistic, pvalue, x2dof, exp = stats.chi2_contingency(crosstab.values)

        dof = min(crosstab.shape[0], crosstab.shape[1]) - 1

        cv = stats.contingency.association(
            crosstab.values, method="cramer", correction=True
        )

        revised_dof = np.min(np.array([dof, 10]))

        thresholds = np.array(self._thresholds[revised_dof])

        strength = self._labels[np.argmax(np.where(thresholds < cv))]

        # Create the result object.
        self._result = CramersV(
            data=crosstab,
            a=self._a,
            b=self._b,
            dof=dof,
            value=cv,
            strength=strength,
            n=len(data),
            x2alpha=self._alpha,
            x2=statistic,
            x2dof=x2dof,
            pvalue=pvalue,
            expected_freq=exp,
        )
