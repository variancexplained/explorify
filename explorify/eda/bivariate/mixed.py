#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Explorify                                                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /explorify/eda/bivariate/mixed.py                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john@variancexplained.com                                                           #
# URL        : https://github.com/variancexplained/explorify                                       #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday June 12th 2024 03:56:32 pm                                                #
# Modified   : Thursday June 13th 2024 03:11:41 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
"""Mixed Categorical / Numeric Bivariate Analysis Module"""
import numpy as np
import pandas as pd
from dependency_injector.wiring import Provide, inject

from explorify.container import VisualizeContainer
from explorify.eda.stats.inferential.centrality import (
    AnovaOneWayTest,
    AnovaOneWayTestResult,
    MannWhitneyUTest,
    MannWhitneyUTestResult,
    WilcoxonSignedRankTest,
    WilcoxonSignedRankTestResult,
)
from explorify.eda.stats.inferential.correlation import (
    PointBiserialCorrelationTest,
    PointBiserialCorrelationTestResult,
)
from explorify.eda.stats.inferential.independence import (
    ChiSquareIndependenceTest,
    ChiSquareIndependenceTestResult,
)
from explorify.eda.stats.inferential.rank import (
    KruskalWallisHTest,
    KruskalWallisHTestResult,
)
from explorify.eda.stats.inferential.variance import LeveneTest, LeveneTestResult
from explorify.eda.visualize.visualizer import Visualizer


# ------------------------------------------------------------------------------------------------ #
class BivariateCategoricalNumericAnalysis:
    """
    A class to perform bivariate analysis between categorical and numeric variables.

    Attributes:
        data (pd.DataFrame): The dataset containing the variables.
        visualizer (Visualizer): An instance of the Visualizer class for creating plots.

    Methods:
        summary_statistics(a: str, b_name: str):
            Computes summary statistics of the numeric variable grouped by the categorical variable.
        anova_test(a: str, b_name: str, alpha: float = 0.05) -> AnovaOneWayTestResult:
            Performs a one-way ANOVA test.
        kruskal_wallis_test(a: str, b_name: str, alpha: float = 0.05) -> KruskalWallisHTestResult:
            Performs a Kruskal-Wallis H test.
        mann_whitney_u_test(a: str, b_name: str, alpha: float = 0.05) -> MannWhitneyUTestResult:
            Performs a Mann-Whitney U test.
        wilcoxon_test(a: str, b_name: str, alpha: float = 0.05) -> WilcoxonSignedRankTestResult:
            Performs a Wilcoxon signed-rank test.
        point_biserial_correlation(a: str, b_name: str, alpha: float = 0.05) -> float:
            Computes the point-biserial correlation coefficient using Pearson's R.
        chi_square_test(a: str, b_name: str, alpha: float = 0.05) -> ChiSquareIndependenceTestResult:
            Performs a Chi-Square test.
        levene_test(a: str, b_name: str, alpha: float = 0.05) -> LeveneTestResult:
            Performs Levene's test for equality of variances.
        box_plot(a: str, b_name: str):
            Creates a box plot of the numeric variable grouped by the categorical variable.
        violin_plot(a: str, b_name: str):
            Creates a violin plot of the numeric variable grouped by the categorical variable.
        effect_size(a: str, b_name: str) -> float:
            Computes the effect size (Eta squared) for the relationship between the categorical and numeric variables.
    """

    @inject
    def __init__(
        self,
        data: pd.DataFrame,
        visualizer: Visualizer = Provide[VisualizeContainer.visualizer],
        anova_test_cls: type[AnovaOneWayTest] = AnovaOneWayTest,
        kruskal_wallis_test_cls: type[KruskalWallisHTest] = KruskalWallisHTest,
        mann_whitney_u_test_cls: type[MannWhitneyUTest] = MannWhitneyUTest,
        wilcoxon_test_cls: type[WilcoxonSignedRankTest] = WilcoxonSignedRankTest,
        point_biserial_correlation_test_cls: type[
            PointBiserialCorrelationTest
        ] = PointBiserialCorrelationTest,
        chi_square_test_cls: type[
            ChiSquareIndependenceTest
        ] = ChiSquareIndependenceTest,
        levene_test_cls: type[LeveneTest] = LeveneTest,
    ):
        """
        Initializes the BivariateCategoricalNumericAnalysis with the given data and variables.

        Args:
            data (pd.DataFrame): The dataset containing the variables.
            visualizer (Visualizer, optional): An instance of the Visualizer class for creating plots. Default provided by dependency injection.
        """
        self._data = data
        self._visualizer = visualizer
        self._anova_test_cls = anova_test_cls
        self._chi_square_test_cls = chi_square_test_cls
        self._kruskal_wallis_test_cls = kruskal_wallis_test_cls
        self._levene_test_cls = levene_test_cls
        self._mann_whitney_u_test_cls = mann_whitney_u_test_cls
        self._point_biserial_correlation_test_cls = point_biserial_correlation_test_cls
        self._wilcoxon_test_cls = wilcoxon_test_cls

    def summary_statistics(self, a_name: str, b_name: str) -> pd.DataFrame:
        """
        Computes summary statistics of the numeric variable grouped by the categorical variable.

        Args:
            a (str): The name of the categorical variable.
            b (str): The name of the numeric variable.

        Returns:
            pd.DataFrame: A DataFrame containing the summary statistics.

        Example:
            >>> analysis = BivariateCategoricalNumericAnalysis(data)
            >>> analysis.summary_statistics('Category', 'Value')
        """
        return self._data.groupby(a_name)[b_name].describe()

    def anova_test(
        self, a_name: str, b_name: str, alpha: float = 0.05
    ) -> AnovaOneWayTestResult:
        """
        Perform one-way ANOVA.

        The one-way ANOVA tests the null hypothesis that two or more groups have the same population mean.
        The test is applied to samples from two or more groups, possibly with differing sizes.

        Args:
            a_name (str): The name of the categorical variable.
            b_name (str): The name of the numeric variable.
            alpha (float): The level of statistical significance for inference. Default = 0.05

        Returns:
            AnovaOneWayTestResult: The result of the ANOVA test.

        Example:
            >>> analysis = BivariateCategoricalNumericAnalysis(data)
            >>> result = analysis.anova_test('Category', 'Value')
            >>> print(result)
        """
        test = self._anova_test_cls(
            a_name=a_name, b_name=b_name, data=self._data, alpha=alpha
        )
        test.run()
        return test.result

    def kruskal_wallis_test(
        self, a_name: str, b_name: str, alpha: float = 0.05
    ) -> KruskalWallisHTestResult:
        """
        Compute the Kruskal-Wallis H-test for independent samples.

        The Kruskal-Wallis H-test tests the null hypothesis that the population median of all of
        the groups are equal. It is a non-parametric version of ANOVA. The test works on 2 or
        more independent samples, which may have different sizes. Note that rejecting the null
        hypothesis does not indicate which of the groups differs. Post hoc comparisons between
        groups are required to determine which groups are different.

        Args:
            a (str): The name of the categorical variable.
            b (str): The name of the numeric variable.
            alpha (float): The level of statistical significance for inference. Default = 0.05

        Returns:
            KruskalWallisHTestResult: The result of the Kruskal-Wallis H test.

        Example:
            >>> analysis = BivariateCategoricalNumericAnalysis(data)
            >>> result = analysis.kruskal_wallis_test('Category', 'Value')
            >>> print(result)
        """
        test = self._kruskal_wallis_test_cls(
            a_name=a_name, b_name=b_name, data=self._data, alpha=alpha
        )
        test.run()
        return test.result

    def mann_whitney_u_test(
        self,
        a_data: np.ndarray,
        a_name: str,
        b_data: np.ndarray,
        b_name: str,
        varname: str,
        alpha: float = 0.05,
    ) -> MannWhitneyUTestResult:
        """
        Perform the Mann-Whitney U rank test on two independent samples.

        The Mann-Whitney U test is a nonparametric test of the null hypothesis that
        the distribution underlying sample x is the same as the distribution underlying
        sample y. It is often used as a test of difference in location between distributions.

        Args:
            a (np.ndarray): Input array
            a_name (str): Name of the group
            b (np.ndarray): Input array
            b (str): The name of the group.
            varname (str): Name of variable of interest
            alpha (float): The level of statistical significance for inference. Default = 0.05

        Returns:
            MannWhitneyUTestResult: The result of the Mann-Whitney U test.

        Example:
            >>> analysis = BivariateCategoricalNumericAnalysis(data)
            >>> result = analysis.mann_whitney_u_test('Category', 'Value')
            >>> print(result)
        """
        test = self._mann_whitney_u_test_cls(
            a_name=a_name,
            a_data=a_data,
            b_name=b_name,
            b_data=b_data,
            varname=varname,
            alpha=alpha,
        )
        test.run()
        return test.result

    def wilcoxon_test(
        self,
        a_name: str,
        a_data: np.ndarray,
        b_name: str,
        b_data: np.ndarray,
        varname: str,
        correction: bool = False,
        alternative: str = "two-sided",
        alpha: float = 0.05,
    ) -> WilcoxonSignedRankTestResult:
        """
        Perform the Wilcoxon signed-rank test.

        The Wilcoxon signed-rank test tests the null hypothesis that two related paired samples come
        from the same distribution. In particular, it tests whether the distribution of the differences
        x - y is symmetric about zero. It is a non-parametric version of the paired T-test.

        Args:
            a_name (str): Name of the first sample
            a_data (np.ndarray): Input array for first sample
            b_name (str): The name of second sample
            b_data (np.ndarray): Input array for second sample
            varname (str): The variable of interest
            correction (bool): If True, apply continuity correction by adjusting the Wilcoxon rank
                statistic by 0.5 towards the mean value when computing the z-statistic if a
                normal approximation is used. Default is False.
            alternative (str): {“two-sided”, “greater”, “less”}, optional
                Defines the alternative hypothesis. Default is 'two-sided'. In the following,
                let d represent the difference between the paired samples: d = x - y if
                both x and y are provided, or d = x otherwise.

                'two-sided': the distribution underlying d is not symmetric about zero.
                'less': the distribution underlying d is stochastically less than a distribution symmetric about zero.
                'greater': the distribution underlying d is stochastically greater than a distribution symmetric about zero.

            alpha (float): The level of statistical significance for inference. Default = 0.05

        Returns:
            WilcoxonSignedRankTestResult: The result of the Wilcoxon signed-rank test.

        Example:
            >>> analysis = BivariateCategoricalNumericAnalysis(data)
            >>> result = analysis.wilcoxon_test('Category', 'Value')
            >>> print(result)
        """
        test = self._wilcoxon_test_cls(
            a_name=a_name,
            a_data=a_data,
            b_name=b_name,
            b_data=b_data,
            varname=varname,
            correction=correction,
            alternative=alternative,
            alpha=alpha,
        )
        test.run()
        return test.result

    def point_biserial_correlation(
        self, a_name: str, b_name: str, alpha: float = 0.05
    ) -> PointBiserialCorrelationTestResult:
        """
        Computes the point-biserial correlation coefficient.
        Uses Pearson's R Correlation Test

        Args:
            a (str): The name of the categorical variable.
            b (str): The name of the numeric variable.
            alpha (float): The level of statistical significance for inference. Default = 0.05

        Returns:
            float: The point-biserial correlation coefficient.

        Example:
            >>> analysis = BivariateCategoricalNumericAnalysis(data)
            >>> r_pb = analysis.point_biserial_correlation('Category', 'Value')
            >>> print(r_pb)
        """
        test = self._point_biserial_correlation_test_cls(
            a_name=a_name,
            b_name=b_name,
            alpha=alpha,
            data=self._data,
        )
        test.run()
        return test.result

    def chi_square_test(
        self, a_name: str, b_name: str, alpha: float = 0.05
    ) -> ChiSquareIndependenceTestResult:
        """
        Performs a Chi-Square test.

        Args:
            a_name (str): The name of the categorical variable.
            b_name (str): The name of the numeric variable.
            alpha (float): The level of statistical significance for inference. Default = 0.05

        Returns:
            ChiSquareIndependenceTestResult: The result of the Chi-Square test.

        Example:
            >>> analysis = BivariateCategoricalNumericAnalysis(data)
            >>> result = analysis.chi_square_test('Category', 'Value')
            >>> print(result)
        """
        test = self._chi_square_test_cls(
            a_name=a_name, b_name=b_name, data=self._data, alpha=alpha
        )
        test.run()
        return test.result

    def levene_test(
        self, a_name: str, b_name: str, alpha: float = 0.05
    ) -> LeveneTestResult:
        """
        Performs Levene's test for equality of variances.

        Args:
            a (str): The name of the categorical variable.
            b (str): The name of the numeric variable.
            alpha (float): The level of statistical significance for inference. Default = 0.05

        Returns:
            LeveneTestResult: The result of Levene's test.

        Example:
            >>> analysis = BivariateCategoricalNumericAnalysis(data)
            >>> result = analysis.levene_test('Category', 'Value')
            >>> print(result)
        """
        test = self._levene_test_cls(
            a_name=a_name, b_name=b_name, data=self._data, alpha=alpha
        )
        test.run()
        return test.result

    def box_plot(self, a_name: str, b_name: str):
        """
        Creates a box plot of the numeric variable grouped by the categorical variable.

        Args:
            a (str): The name of the categorical variable.
            b (str): The name of the numeric variable.

        Example:
            >>> analysis = BivariateCategoricalNumericAnalysis(data)
            >>> analysis.box_plot('Category', 'Value')
        """
        self._visualizer.boxplot(x=a_name, y=b_name, data=self._data)

    def violin_plot(self, a_name: str, b_name: str):
        """
        Creates a violin plot of the numeric variable grouped by the categorical variable.

        Args:
            a (str): The name of the categorical variable.
            b (str): The name of the numeric variable.

        Example:
            >>> analysis = BivariateCategoricalNumericAnalysis(data)
            >>> analysis.violin_plot('Category', 'Value')
        """
        self._visualizer.violinplot(x=a_name, y=b_name, data=self._data)

    def effect_size(self, a_name: str, b_name: str) -> float:
        """
        Computes the effect size (Eta squared) for the relationship between the categorical and numeric variables.

        Args:
            a (str): The name of the categorical variable.
            b (str): The name of the numeric variable.

        Returns:
            float: The Eta squared effect size.

        Example:
            >>> analysis = BivariateCategoricalNumericAnalysis(data)
            >>> eta_sq = analysis.effect_size('Category', 'Value')
            >>> print(eta_sq)
        """
        group_means = self._data.groupby(a_name)[b_name].mean()
        overall_mean = self._data[b_name].mean()
        ss_between = sum(
            self._data.groupby(a_name).size() * (group_means - overall_mean) ** 2
        )
        ss_total = sum((self._data[b_name] - overall_mean) ** 2)
        eta_squared = ss_between / ss_total
        return eta_squared
