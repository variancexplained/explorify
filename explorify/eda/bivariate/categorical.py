#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Explorify                                                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /explorify/eda/bivariate/categorical.py                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john@variancexplained.com                                                           #
# URL        : https://github.com/variancexplained/explorify                                       #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday June 9th 2024 10:52:53 am                                                    #
# Modified   : Sunday June 9th 2024 03:20:14 pm                                                    #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
from abc import ABC
from typing import Type

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mutual_info_score

from explorify.eda.stats.inferential.association import (
    CramersVAnalysis,
    KendallsTauAnalysis,
)
from explorify.eda.stats.inferential.base import StatTestResult
from explorify.eda.stats.inferential.correlation import SpearmanCorrelationTest
from explorify.eda.stats.inferential.independence import ChiSquareIndependenceTest


# ------------------------------------------------------------------------------------------------ #
class CategoricalBivariateAnalysis(ABC):
    """
    Base class for categorical-to-categorical bivariate analysis.

    Args:
        _data (pd.DataFrame): The input data containing the variables.
        cramers_analysis_cls (Type[CramersVAnalysis]): The class for conducting Cramér's V analysis.
        chisquare_analysis_cls (Type[ChiSquareAnalysis]): The class for conducting chi-square analysis.

    Methods:
        [describe methods here.]


    """

    def __init__(
        self,
        data: pd.DataFrame,
        cramers_analysis_cls: Type[CramersVAnalysis] = CramersVAnalysis,
        chisquare_test_cls: Type[ChiSquareIndependenceTest] = ChiSquareIndependenceTest,
    ):
        """
        Initializes the CategoricalBivariateAnalysis instance."""
        self._data = data
        self.cramers_analysis_cls = cramers_analysis_cls
        self.chisquare_test_cls = chisquare_test_cls

    def validate_input(self, var1: str, var2: str) -> None:
        """
        Validates the input variables.

        Parameters
        ----------
        var1 : str
            The name of the first categorical variable.
        var2 : str
            The name of the second categorical variable.

        Raises
        ------
        ValueError
            If the input variables are not in the DataFrame or are numeric.
        """
        if var1 not in self._data.columns or var2 not in self._data.columns:
            raise ValueError(
                f"Variables '{var1}' and/or '{var2}' are not in the DataFrame."
            )

        if pd.api.types.is_numeric_dtype(self._data[var1]):
            raise ValueError(f"Variable '{var1}' is numeric and not categorical.")

        if pd.api.types.is_numeric_dtype(self._data[var2]):
            raise ValueError(f"Variable '{var2}' is numeric and not categorical.")

    def contingency_table(self, var1: str, var2: str) -> pd.DataFrame:
        """
        Computes the contingency table for two categorical variables.

        Parameters
        ----------
        var1 : str
            The name of the first categorical variable.
        var2 : str
            The name of the second categorical variable.

        Returns
        -------
        pd.DataFrame
            Contingency table of the two variables.
        """
        self.validate_input(var1, var2)
        return pd.crosstab(self._data[var1], self._data[var2])

    def chi_square(self, var1: str, var2: str) -> StatTestResult:
        """
        Computes the chi-square statistic for two categorical variables.

        Parameters
        ----------
        var1 : str
            The name of the first categorical variable.
        var2 : str
            The name of the second categorical variable.

        Returns
        -------
        tuple[float, float]
            Chi-square statistic and p-value of the test.
        """
        self.validate_input(var1, var2)
        analysis = self.chisquare_test_cls(data=self._data, a=var1, b=var2)
        analysis.run()
        return analysis.result

    def cramers_v(self, var1: str, var2: str) -> StatTestResult:
        """
        Computes Cramér's V statistic for two categorical variables.

        Parameters
        ----------
        var1 : str
            The name of the first categorical variable.
        var2 : str
            The name of the second categorical variable.

        Returns
        -------
        float
            Cramér's V statistic.
        """
        self.validate_input(var1, var2)
        analysis = self.cramers_analysis_cls(self._data, a=var1, b=var2)
        analysis.run()
        return analysis.result

    def mutual_information(self, var1: str, var2: str) -> float:
        """
        Computes the mutual information between two categorical variables.

        Parameters
        ----------
        var1 : str
            The name of the first categorical variable.
        var2 : str
            The name of the second categorical variable.

        Returns
        -------
        float
            Mutual information score.
        """
        self.validate_input(var1, var2)
        return mutual_info_score(self._data[var1], self._data[var2])


# ------------------------------------------------------------------------------------------------ #
#                             NOMINAL / NOMINAL BIVARIATE ANALYSIS                                 #
# ------------------------------------------------------------------------------------------------ #
class NominalNominalBivariateAnalysis(CategoricalBivariateAnalysis):
    """
    Class for nominal-to-nominal bivariate analysis.

    Inherits from CategoricalBivariateAnalysis.

    Args:
        _data (pd.DataFrame): The input data containing the variables.

    Attributes:
        _data (pd.DataFrame): The input data containing the variables.

    Methods:
        phi_coefficient(var1: str, var2: str) -> float:
            Computes the Phi coefficient for two nominal variables (for 2x2 tables).
        contingency_coefficient(var1: str, var2: str) -> float:
            Computes the contingency coefficient for two nominal variables.
        lambda_coefficient(var1: str, var2: str) -> float:
            Computes the lambda coefficient for two nominal variables.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initializes the NominalNominalBivariateAnalysis instance.

        Args:
            _data (pd.DataFrame): The input data containing the variables.
        """
        super().__init__(data=data)

    def phi_coefficient(self, var1: str, var2: str) -> float:
        """
        Computes the Phi coefficient for two nominal variables (for 2x2 tables).

        Args:
            var1 (str): The name of the first nominal variable.
            var2 (str): The name of the second nominal variable.

        Returns:
            float: Phi coefficient.
        """
        self.validate_input(var1, var2)
        contingency_table = self.contingency_table(var1, var2)
        if contingency_table.shape != (2, 2):
            raise ValueError(
                "Phi coefficient can only be computed for 2x2 contingency tables."
            )

        chi2, _, _, _ = stats.chi2_contingency(contingency_table)
        n = self._data.shape[0]
        return np.sqrt(chi2 / n)

    def contingency_coefficient(self, var1: str, var2: str) -> float:
        """
        Computes the contingency coefficient for two nominal variables.

        Args:
            var1 (str): The name of the first nominal variable.
            var2 (str): The name of the second nominal variable.

        Returns:
            float: Contingency coefficient.
        """
        self.validate_input(var1, var2)
        contingency_table = self.contingency_table(var1, var2)
        chi2, _, _, _ = stats.chi2_contingency(contingency_table)
        n = self._data.shape[0]
        return np.sqrt(chi2 / (chi2 + n))

    def lambda_coefficient(self, var1: str, var2: str) -> float:
        """
        Computes the lambda coefficient for two nominal variables.

        Args:
            var1 (str): The name of the first nominal variable.
            var2 (str): The name of the second nominal variable.

        Returns:
            float: Lambda coefficient.
        """
        self.validate_input(var1, var2)
        contingency_table = self.contingency_table(var1, var2)
        chi2, _, _, _ = stats.chi2_contingency(contingency_table)
        n = self._data.shape[0]
        return np.sqrt(chi2 / (n * min(contingency_table.shape)))


# ------------------------------------------------------------------------------------------------ #
#                             NOMINAL / ORDINAL BIVARIATE ANALYSIS                                 #
# ------------------------------------------------------------------------------------------------ #
class NominalOrdinalBivariateAnalysis(CategoricalBivariateAnalysis):
    """
    Class for nominal-to-ordinal bivariate analysis.

    Inherits from CategoricalBivariateAnalysis.

    Args:
        _data (pd.DataFrame): The input data containing the variables.

    Attributes:
        _data (pd.DataFrame): The input data containing the variables.

    Methods:
        gamma(var1: str, var2: str) -> float:
            Computes the Gamma coefficient for a nominal and an ordinal variable.
        lambda_coefficient(var1: str, var2: str) -> float:
            Computes the lambda coefficient for a nominal and an ordinal variable.
        theil_u(var1: str, var2: str) -> float:
            Computes Theil's U coefficient for a nominal and an ordinal variable.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initializes the NominalOrdinalBivariateAnalysis instance.

        Args:
            _data (pd.DataFrame): The input data containing the variables.
        """
        super().__init__(data=data)

    def contingency_table(self, var1: str, var2: str) -> pd.DataFrame:
        """
        Computes the contingency table for a nominal variable and an ordinal variable.

        Args:
            var1 (str): The name of the nominal variable.
            var2 (str): The name of the ordinal variable.

        Returns:
            pd.DataFrame: Contingency table of the two variables.
        """
        # Sort the ordinal variable values
        sorted_ordinal_values = self._data[var2].sort_values()

        # Create a contingency table using the nominal variable and sorted ordinal values
        return pd.crosstab(self._data[var1], sorted_ordinal_values)

    def gamma(self, var1: str, var2: str) -> float:
        """
        Computes the Gamma coefficient for a nominal and an ordinal variable.

        Args:
            var1 (str): The name of the nominal variable.
            var2 (str): The name of the ordinal variable. Note, ordinal
                variables must be in lexicographical order. For instance
                "0_low_income", "1_moderate_income", etc...

        Returns:
            float: Gamma coefficient.
        """
        self.validate_input(var1, var2)
        contingency_table = self.contingency_table(var1, var2)

        # Compute the ranks for the ordinal variable
        ranks = self._data[var2].rank(method="dense")

        # Calculate the differences in ranks for concordant and discordant pairs
        concordant_pairs = 0
        discordant_pairs = 0
        for i in range(len(self._data)):
            for j in range(i + 1, len(self._data)):
                if self._data[var1][i] != self._data[var1][j]:
                    if (ranks[i] - ranks[j]) * (
                        contingency_table.loc[self._data[var1][i], self._data[var2][j]]
                        - contingency_table.loc[
                            self._data[var1][j], self._data[var2][i]
                        ]
                    ) > 0:
                        concordant_pairs += 1
                    else:
                        discordant_pairs += 1

        return (concordant_pairs - discordant_pairs) / (
            concordant_pairs + discordant_pairs
        )

    def lambda_coefficient(self, var1: str, var2: str) -> float:
        """
        Computes the lambda coefficient for a nominal and an ordinal variable.

        Args:
            var1 (str): The name of the nominal variable.
            var2 (str): The name of the ordinal variable. Note, ordinal
                variables must be in lexicographical order. For instance
                "0_low_income", "1_moderate_income", etc...

        Returns:
            float: Lambda coefficient.
        """
        self.validate_input(var1, var2)
        contingency_table = self.contingency_table(var1, var2)
        n = contingency_table.sum().sum()  # Total number of observations

        # Calculate the marginal proportions
        row_marginals = contingency_table.sum(axis=1) / n
        col_marginals = contingency_table.sum(axis=0) / n

        # Calculate the observed agreement
        observed_agreement = 0
        for i in range(contingency_table.shape[0]):
            for j in range(contingency_table.shape[1]):
                observed_agreement += (
                    contingency_table.iloc[i, j] * row_marginals[i] * col_marginals[j]
                )

        # Calculate the expected agreement under independence assumption
        expected_agreement = 0
        for i in range(contingency_table.shape[0]):
            for j in range(contingency_table.shape[1]):
                expected_agreement += row_marginals[i] * col_marginals[j]

        return (observed_agreement - expected_agreement) / (1 - expected_agreement)

    def theil_u(self, var1: str, var2: str) -> float:
        """
        Computes Theil's U coefficient for a nominal and an ordinal variable.

        Args:
            var1 (str): The name of the nominal variable.
            var2 (str): The name of the ordinal variable. Note, ordinal
                variables must be in lexicographical order. For instance
                "0_low_income", "1_moderate_income", etc...

        Returns:
            float: Theil's U coefficient.
        """
        self.validate_input(var1, var2)
        # Calculate the marginal probabilities
        p_x = self._data[var1].value_counts(normalize=True)
        p_y_given_x = self._data.groupby(var1)[var2].value_counts(normalize=True)

        # Compute Theil's U coefficient
        u = 0
        for x in p_x.index:
            for y in p_y_given_x[x].index:
                p_xy = p_y_given_x[x][y]
                u += p_xy * np.log(p_xy / p_x[x])

        return u


# ------------------------------------------------------------------------------------------------ #
#                             ORDINAL / ORDINAL BIVARIATE ANALYSIS                                 #
# ------------------------------------------------------------------------------------------------ #
class OrdinalOrdinalBivariateAnalysis(CategoricalBivariateAnalysis):
    """
    Class for ordinal-to-ordinal bivariate analysis.

    Inherits from CategoricalBivariateAnalysis.

    Args:
        _data (pd.DataFrame): The input data containing the variables.

    Attributes:
        _data (pd.DataFrame): The input data containing the variables.

    Methods:
        contingency_table(var1: str, var2: str) -> pd.DataFrame:
            Computes the contingency table for two ordinal variables.
        cramer_v(var1: str, var2: str) -> float:
            Computes Cramér's V statistic for two ordinal variables.
        gamma(var1: str, var2: str) -> float:
            Computes the Gamma coefficient for two ordinal variables.
        kendalls_tau(var1: str, var2: str) -> float:
            Computes Kendall's Tau coefficient for two ordinal variables.
        spearmans_rank(var1: str, var2: str) -> float:
            Computes Spearman's Rank Correlation coefficient for two ordinal variables.
        mutual_information(var1: str, var2: str) -> float:
            Computes Mutual Information adjusted for ordinal data for two ordinal variables.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        kendalls_tau_cls: Type[KendallsTauAnalysis] = KendallsTauAnalysis,
        spearmans_rank_cls: Type[SpearmanCorrelationTest] = SpearmanCorrelationTest,
    ):
        """
        Initializes the OrdinalOrdinalBivariateAnalysis instance.

        Args:
            _data (pd.DataFrame): The input data containing the variables.
        """
        super().__init__(data=data)
        self.kendalls_tau_cls = kendalls_tau_cls
        self.spearmans_rank_cls = spearmans_rank_cls

    def contingency_table(self, var1: str, var2: str) -> pd.DataFrame:
        """
        Computes the contingency table for two ordinal variables.

        Args:
            var1 (str): The name of the first ordinal variable.
            var2 (str): The name of the second ordinal variable.

        Returns:
            pd.DataFrame: Contingency table of the two variables.
        """
        # Sort both ordinal variables
        sorted_var1 = self._data[var1].sort_values()
        sorted_var2 = self._data[var2].sort_values()

        # Create a contingency table using the sorted ordinal variables
        return pd.crosstab(sorted_var1, sorted_var2)

    def cramer_v(self, var1: str, var2: str) -> float:
        """
        Computes Cramér's V statistic for two ordinal variables.

        Args:
            var1 (str): The name of the first ordinal variable.This assumes that variable values
                have lexicographical order
            var2 (str): The name of the second ordinal variable.This assumes that variable values
                have lexicographical order

        Returns:
            float: Cramér's V statistic.
        """
        self.validate_input(var1, var2)
        analysis = self.cramers_analysis_cls(
            data=self._data, a=var1, b=var2, ordinal_a=True, ordinal_b=True
        )
        analysis.run()
        return analysis.result

    def gamma(self, var1: str, var2: str) -> float:
        """
        Computes the Gamma coefficient for two ordinal variables.

        Args:
            var1 (str): The name of the first ordinal variable.
            var2 (str): The name of the second ordinal variable. Both ordinal variables must be sorted.

        Returns:
            float: Gamma coefficient.
        """
        self.validate_input(var1, var2)

        # Sort both ordinal variables
        sorted_var1 = self._data[var1].sort_values()
        sorted_var2 = self._data[var2].sort_values()

        # Compute ranks for both ordinal variables
        ranks_var1 = sorted_var1.rank(method="dense")
        ranks_var2 = sorted_var2.rank(method="dense")

        # Calculate the differences in ranks for concordant and discordant pairs
        concordant_pairs = 0
        discordant_pairs = 0
        for i in range(len(self._data)):
            for j in range(i + 1, len(self._data)):
                if (
                    sorted_var1.iloc[i] != sorted_var1.iloc[j]
                    and sorted_var2.iloc[i] != sorted_var2.iloc[j]
                ):  # pragma: no cover
                    if (ranks_var1.iloc[i] - ranks_var1.iloc[j]) * (
                        ranks_var2.iloc[i] - ranks_var2.iloc[j]
                    ) > 0:  # pragma: no cover
                        concordant_pairs += 1
                    else:
                        discordant_pairs += 1

        return (concordant_pairs - discordant_pairs) / (
            concordant_pairs + discordant_pairs
        )

    def kendalls_tau(self, var1: str, var2: str) -> float:
        """
        Computes Kendall's Tau coefficient for two ordinal variables.

        Args:
            var1 (str): The name of the first ordinal variable. This assumes that variable values
                have lexicographical order
            var2 (str): The name of the second ordinal variable.This assumes that variable values
                have lexicographical order

        Returns:
            float: Kendall's Tau coefficient.
        """
        self.validate_input(var1, var2)
        analysis = self.kendalls_tau_cls(
            data=self._data, a=var1, b=var2, ordinal_a=True, ordinal_b=True
        )
        analysis.run()
        return analysis.result

    def spearmans_rank(self, var1: str, var2: str) -> float:
        """
        Computes Spearman's Rank Correlation coefficient for two ordinal variables.

        Args:
            var1 (str): The name of the first ordinal variable.
            var2 (str): The name of the second ordinal variable.

        Returns:
            float: Spearman's Rank Correlation coefficient.
        """
        self.validate_input(var1, var2)
        analysis = self.spearmans_rank_cls(data=self._data, a=var1, b=var2)
        analysis.run()
        return analysis.result

    def mutual_information(self, var1: str, var2: str) -> float:
        """
        Computes Mutual Information adjusted for ordinal data for two ordinal variables.

        Args:
            var1 (str): The name of the first ordinal variable. This assumes that variable values
                have lexicographical order
            var2 (str): The name of the second ordinal variable. This assumes that variable values
                have lexicographical order

        Returns:
            float: Mutual Information adjusted for ordinal data.
        """
        self.validate_input(var1, var2)

        # Assign ranks to the ordinal values
        ranks_var1 = self._data[var1].rank(method="dense")
        ranks_var2 = self._data[var2].rank(method="dense")

        # Compute mutual information using ranked variables
        mutual_info = mutual_info_score(ranks_var1, ranks_var2)

        return mutual_info
