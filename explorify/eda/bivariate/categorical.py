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
# Modified   : Thursday June 13th 2024 11:29:53 am                                                 #
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
    KendallsTauTest,
    KendallsTauTestResult,
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
        self._cramers_analysis_cls = cramers_analysis_cls
        self._chisquare_test_cls = chisquare_test_cls

    def validate_input(self, a_name: str, b_name: str) -> None:
        """
        Validates the input variables.

        Parameters
        ----------
        a_name : str
            The name of the first categorical variable.
        b_name : str
            The name of the second categorical variable.

        Raises
        ------
        ValueError
            If the input variables are not in the DataFrame or are numeric.
        """
        if a_name not in self._data.columns or b_name not in self._data.columns:
            raise ValueError(
                f"Variables '{a_name}' and/or '{b_name}' are not in the DataFrame."
            )

        if pd.api.types.is_numeric_dtype(self._data[a_name]):
            raise ValueError(f"Variable '{a_name}' is numeric and not categorical.")

        if pd.api.types.is_numeric_dtype(self._data[b_name]):
            raise ValueError(f"Variable '{b_name}' is numeric and not categorical.")

    def contingency_table(self, a_name: str, b_name: str) -> pd.DataFrame:
        """
        Computes the contingency table for two categorical variables.

        Parameters
        ----------
        a_name : str
            The name of the first categorical variable.
        b_name : str
            The name of the second categorical variable.

        Returns
        -------
        pd.DataFrame
            Contingency table of the two variables.
        """
        self.validate_input(a_name, b_name)
        return pd.crosstab(self._data[a_name], self._data[b_name])

    def chi_square(self, a_name: str, b_name: str) -> StatTestResult:
        """
        Computes the chi-square statistic for two categorical variables.

        Parameters
        ----------
        a_name : str
            The name of the first categorical variable.
        b_name : str
            The name of the second categorical variable.

        Returns
        -------
        tuple[float, float]
            Chi-square statistic and p-value of the test.
        """
        self.validate_input(a_name, b_name)
        analysis = self._chisquare_test_cls(
            data=self._data, a_name=a_name, b_name=b_name
        )
        analysis.run()
        return analysis.result

    def cramers_v(self, a_name: str, b_name: str) -> StatTestResult:
        """
        Computes Cramér's V statistic for two categorical variables.

        Parameters
        ----------
        a_name : str
            The name of the first categorical variable.
        b_name : str
            The name of the second categorical variable.

        Returns
        -------
        float
            Cramér's V statistic.
        """
        self.validate_input(a_name, b_name)
        analysis = self._cramers_analysis_cls(self._data, a_name=a_name, b_name=b_name)
        analysis.run()
        return analysis.result

    def mutual_information(self, a_name: str, b_name: str) -> float:
        """
        Computes the mutual information between two categorical variables.

        Parameters
        ----------
        a_name : str
            The name of the first categorical variable.
        b_name : str
            The name of the second categorical variable.

        Returns
        -------
        float
            Mutual information score.
        """
        self.validate_input(a_name, b_name)
        return mutual_info_score(self._data[a_name], self._data[b_name])


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
        phi_coefficient(a_name: str, b_name: str) -> float:
            Computes the Phi coefficient for two nominal variables (for 2x2 tables).
        contingency_coefficient(a_name: str, b_name: str) -> float:
            Computes the contingency coefficient for two nominal variables.
        lambda_coefficient(a_name: str, b_name: str) -> float:
            Computes the lambda coefficient for two nominal variables.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initializes the NominalNominalBivariateAnalysis instance.

        Args:
            _data (pd.DataFrame): The input data containing the variables.
        """
        super().__init__(data=data)

    def phi_coefficient(self, a_name: str, b_name: str) -> float:
        """
        Computes the Phi coefficient for two nominal variables (for 2x2 tables).

        Args:
            a_name (str): The name of the first nominal variable.
            b_name (str): The name of the second nominal variable.

        Returns:
            float: Phi coefficient.
        """
        self.validate_input(a_name, b_name)
        contingency_table = self.contingency_table(a_name, b_name)
        if contingency_table.shape != (2, 2):
            raise ValueError(
                "Phi coefficient can only be computed for 2x2 contingency tables."
            )

        chi2, _, _, _ = stats.chi2_contingency(contingency_table)
        n = self._data.shape[0]
        return np.sqrt(chi2 / n)

    def contingency_coefficient(self, a_name: str, b_name: str) -> float:
        """
        Computes the contingency coefficient for two nominal variables.

        Args:
            a_name (str): The name of the first nominal variable.
            b_name (str): The name of the second nominal variable.

        Returns:
            float: Contingency coefficient.
        """
        self.validate_input(a_name, b_name)
        contingency_table = self.contingency_table(a_name, b_name)
        chi2, _, _, _ = stats.chi2_contingency(contingency_table)
        n = self._data.shape[0]
        return np.sqrt(chi2 / (chi2 + n))

    def lambda_coefficient(self, a_name: str, b_name: str) -> float:
        """
        Computes the lambda coefficient for two nominal variables.

        Args:
            a_name (str): The name of the first nominal variable.
            b_name (str): The name of the second nominal variable.

        Returns:
            float: Lambda coefficient.
        """
        self.validate_input(a_name, b_name)
        contingency_table = self.contingency_table(a_name, b_name)
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
        gamma(a_name: str, b_name: str) -> float:
            Computes the Gamma coefficient for a nominal and an ordinal variable.
        lambda_coefficient(a_name: str, b_name: str) -> float:
            Computes the lambda coefficient for a nominal and an ordinal variable.
        theil_u(a_name: str, b_name: str) -> float:
            Computes Theil's U coefficient for a nominal and an ordinal variable.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initializes the NominalOrdinalBivariateAnalysis instance.

        Args:
            _data (pd.DataFrame): The input data containing the variables.
        """
        super().__init__(data=data)

    def contingency_table(self, a_name: str, b_name: str) -> pd.DataFrame:
        """
        Computes the contingency table for a nominal variable and an ordinal variable.

        Args:
            a_name (str): The name of the nominal variable.
            b_name (str): The name of the ordinal variable.

        Returns:
            pd.DataFrame: Contingency table of the two variables.
        """
        # Sort the ordinal variable values
        sorted_ordinal_values = self._data[b_name].sort_values()

        # Create a contingency table using the nominal variable and sorted ordinal values
        return pd.crosstab(self._data[a_name], sorted_ordinal_values)

    def gamma(self, a_name: str, b_name: str) -> float:
        """
        Computes the Gamma coefficient for a nominal and an ordinal variable.

        Args:
            a_name (str): The name of the nominal variable.
            b_name (str): The name of the ordinal variable. Note, ordinal
                variables must be in lexicographical order. For instance
                "0_low_income", "1_moderate_income", etc...

        Returns:
            float: Gamma coefficient.
        """
        self.validate_input(a_name, b_name)
        contingency_table = self.contingency_table(a_name, b_name)

        # Compute the ranks for the ordinal variable
        ranks = self._data[b_name].rank(method="dense")

        # Calculate the differences in ranks for concordant and discordant pairs
        concordant_pairs = 0
        discordant_pairs = 0
        for i in range(len(self._data)):
            for j in range(i + 1, len(self._data)):
                if self._data[a_name][i] != self._data[a_name][j]:
                    if (ranks[i] - ranks[j]) * (
                        contingency_table.loc[
                            self._data[a_name][i], self._data[b_name][j]
                        ]
                        - contingency_table.loc[
                            self._data[a_name][j], self._data[b_name][i]
                        ]
                    ) > 0:
                        concordant_pairs += 1
                    else:
                        discordant_pairs += 1

        return (concordant_pairs - discordant_pairs) / (
            concordant_pairs + discordant_pairs
        )

    def lambda_coefficient(self, a_name: str, b_name: str) -> float:
        """
        Computes the lambda coefficient for a nominal and an ordinal variable.

        Args:
            a_name (str): The name of the nominal variable.
            b_name (str): The name of the ordinal variable. Note, ordinal
                variables must be in lexicographical order. For instance
                "0_low_income", "1_moderate_income", etc...

        Returns:
            float: Lambda coefficient.
        """
        self.validate_input(a_name, b_name)
        contingency_table = self.contingency_table(a_name, b_name)
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

    def theil_u(self, a_name: str, b_name: str) -> float:
        """
        Computes Theil's U coefficient for a nominal and an ordinal variable.

        Args:
            a_name (str): The name of the nominal variable.
            b_name (str): The name of the ordinal variable. Note, ordinal
                variables must be in lexicographical order. For instance
                "0_low_income", "1_moderate_income", etc...

        Returns:
            float: Theil's U coefficient.
        """
        self.validate_input(a_name, b_name)
        # Calculate the marginal probabilities
        p_x = self._data[a_name].value_counts(normalize=True)
        p_y_given_x = self._data.groupby(a_name)[b_name].value_counts(normalize=True)

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
        contingency_table(a_name: str, b_name: str) -> pd.DataFrame:
            Computes the contingency table for two ordinal variables.
        cramer_v(a_name: str, b_name: str) -> float:
            Computes Cramér's V statistic for two ordinal variables.
        gamma(a_name: str, b_name: str) -> float:
            Computes the Gamma coefficient for two ordinal variables.
        kendalls_tau(a_name: str, b_name: str) -> float:
            Computes Kendall's Tau coefficient for two ordinal variables.
        spearmans_rank(a_name: str, b_name: str) -> float:
            Computes Spearman's Rank Correlation coefficient for two ordinal variables.
        mutual_information(a_name: str, b_name: str) -> float:
            Computes Mutual Information adjusted for ordinal data for two ordinal variables.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        kendalls_tau_cls: Type[KendallsTauTest] = KendallsTauTest,
        spearmans_rank_cls: Type[SpearmanCorrelationTest] = SpearmanCorrelationTest,
    ):
        """
        Initializes the OrdinalOrdinalBivariateAnalysis instance.

        Args:
            _data (pd.DataFrame): The input data containing the variables.
        """
        super().__init__(data=data)
        self._kendalls_tau_cls = kendalls_tau_cls
        self.spearmans_rank_cls = spearmans_rank_cls

    def contingency_table(self, a_name: str, b_name: str) -> pd.DataFrame:
        """
        Computes the contingency table for two ordinal variables.

        Args:
            a_name (str): The name of the first ordinal variable.
            b_name (str): The name of the second ordinal variable.

        Returns:
            pd.DataFrame: Contingency table of the two variables.
        """
        # Sort both ordinal variables
        sorted_a_name = self._data[a_name].sort_values()
        sorted_b_name = self._data[b_name].sort_values()

        # Create a contingency table using the sorted ordinal variables
        return pd.crosstab(sorted_a_name, sorted_b_name)

    def cramer_v(
        self, a_name: str, b_name: str, ordinal_a: bool = False, ordinal_b: bool = False
    ) -> float:
        """
        Computes Cramér's V statistic for two ordinal variables.

        Args:
            a_name (str): The name of the first ordinal variable.This assumes that variable values
                have lexicographical order
            b_name (str): The name of the second ordinal variable.This assumes that variable values
                have lexicographical order
            ordinal_a (bool): True, if a_name variable is ordinal. False otherwise.
            ordinal_b (bool): True, if b_name variable is ordinal. False otherwise.

        Returns:
            float: Cramér's V statistic.
        """
        self.validate_input(a_name, b_name)
        analysis = self._cramers_analysis_cls(
            data=self._data,
            a_name=a_name,
            b_name=b_name,
            ordinal_a=ordinal_a,
            ordinal_b=ordinal_b,
        )
        analysis.run()
        return analysis.result

    def gamma(self, a_name: str, b_name: str) -> float:
        """
        Computes the Gamma coefficient for two ordinal variables.

        Args:
            a_name (str): The name of the first ordinal variable.
            b_name (str): The name of the second ordinal variable. Both ordinal variables must be sorted.

        Returns:
            float: Gamma coefficient.
        """
        self.validate_input(a_name, b_name)

        # Sort both ordinal variables
        sorted_a_name = self._data[a_name].sort_values()
        sorted_b_name = self._data[b_name].sort_values()

        # Compute ranks for both ordinal variables
        ranks_a_name = sorted_a_name.rank(method="dense")
        ranks_b_name = sorted_b_name.rank(method="dense")

        # Calculate the differences in ranks for concordant and discordant pairs
        concordant_pairs = 0
        discordant_pairs = 0
        for i in range(len(self._data)):
            for j in range(i + 1, len(self._data)):
                if (
                    sorted_a_name.iloc[i] != sorted_a_name.iloc[j]
                    and sorted_b_name.iloc[i] != sorted_b_name.iloc[j]
                ):  # pragma: no cover
                    if (ranks_a_name.iloc[i] - ranks_a_name.iloc[j]) * (
                        ranks_b_name.iloc[i] - ranks_b_name.iloc[j]
                    ) > 0:  # pragma: no cover
                        concordant_pairs += 1
                    else:
                        discordant_pairs += 1

        return (concordant_pairs - discordant_pairs) / (
            concordant_pairs + discordant_pairs
        )

    def kendalls_tau(
        self, a_name: str, b_name: str, ordinal_a: bool = False, ordinal_b: bool = False
    ) -> KendallsTauTestResult:
        """
        Computes Kendall's Tau coefficient for two ordinal variables.

        Args:
            a_name (str): The name of the first ordinal variable. This assumes that variable values
                have lexicographical order
            b_name (str): The name of the second ordinal variable.This assumes that variable values
                have lexicographical order

        Returns:
            float: Kendall's Tau coefficient.
        """
        self.validate_input(a_name, b_name)
        analysis = self._kendalls_tau_cls(
            data=self._data,
            a_name=a_name,
            b_name=b_name,
            ordinal_a=ordinal_a,
            ordinal_b=ordinal_b,
        )
        analysis.run()
        return analysis.result

    def spearmans_rank(self, a_name: str, b_name: str) -> float:
        """
        Computes Spearman's Rank Correlation coefficient for two ordinal variables.

        Args:
            a_name (str): The name of the first ordinal variable.
            b_name (str): The name of the second ordinal variable.

        Returns:
            float: Spearman's Rank Correlation coefficient.
        """
        self.validate_input(a_name, b_name)
        analysis = self.spearmans_rank_cls(
            data=self._data, a_name=a_name, b_name=b_name
        )
        analysis.run()
        return analysis.result

    def mutual_information(self, a_name: str, b_name: str) -> float:
        """
        Computes Mutual Information adjusted for ordinal data for two ordinal variables.

        Args:
            a_name (str): The name of the first ordinal variable. This assumes that variable values
                have lexicographical order
            b_name (str): The name of the second ordinal variable. This assumes that variable values
                have lexicographical order

        Returns:
            float: Mutual Information adjusted for ordinal data.
        """
        self.validate_input(a_name, b_name)

        # Assign ranks to the ordinal values
        ranks_a_name = self._data[a_name].rank(method="dense")
        ranks_b_name = self._data[b_name].rank(method="dense")

        # Compute mutual information using ranked variables
        mutual_info = mutual_info_score(ranks_a_name, ranks_b_name)

        return mutual_info
