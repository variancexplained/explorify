#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Explorify                                                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /explorify/eda/multivariate/analysis.py                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john@variancexplained.com                                                           #
# URL        : https://github.com/variancexplained/explorify                                       #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday June 13th 2024 08:37:22 pm                                                 #
# Modified   : Thursday June 13th 2024 08:49:36 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
"""Multivariate Analaysis Module"""
import numpy as np
import pandas as pd
from dependency_injector.wiring import Provide, inject
from statsmodels.stats.outliers_influence import variance_inflation_factor

from explorify.container import VisualizeContainer
from explorify.eda.multivariate.base import BaseAnalyzer
from explorify.eda.visualize.visualizer import Visualizer


# ------------------------------------------------------------------------------------------------ #
class VIFAnalyzer(BaseAnalyzer):
    """
    A class to compute Variance Inflation Factor (VIF) for multivariate analysis.

    Attributes:
        _visualizer (Visualizer): Instance of Visualizer for plotting.

    Methods:
        analyze(**kwargs) -> pd.DataFrame:
            Computes VIF for each feature.

        plot(**kwargs) -> None:
            Plots the VIF values.

    Notes:
        This class requires a Visualizer instance provided through dependency injection.

    Raises:
        ValueError: If the provided data is not suitable for VIF analysis.

    """

    @inject
    def __init__(
        self,
        visualizer: Visualizer = Provide[VisualizeContainer.visualizer],
    ) -> None:
        """
        Initializes the VIFAnalyzer.

        Args:
            visualizer (Visualizer): Instance of Visualizer for plotting.
        """
        super().__init__()
        self._visualizer = visualizer

    def analyze(self, features: list, **kwargs) -> pd.DataFrame:
        """
        Computes Variance Inflation Factor (VIF) for each feature.

        Args:
            features (list): Features to include in the VIF calculation.
            **kwargs: Additional arguments for computing VIF.

        Returns:
            pd.DataFrame: VIF values for each feature.
        """
        X = self._data.select_dtypes(include=[np.number]).dropna()
        vif_data = pd.DataFrame()
        vif_data["Feature"] = features
        vif_data["VIF"] = [
            variance_inflation_factor(X.values, i) for i in range(X.shape[1])
        ]
        return vif_data

    def plot(
        self,
        palette: str = "Blues_r",
        title: str = "Variance Inflation Factor (VIF)",
        **kwargs
    ) -> None:
        """
        Plots the VIF values.

        Args:
            **kwargs: Additional arguments for plotting (optional).
        """
        vif_data = self.analyze()
        title = title
        self._visualizer.barplot(
            x="Feature", y="VIF", data=vif_data, palette=palette, title=title, **kwargs
        )


# ------------------------------------------------------------------------------------------------ #
class CorrelationAnalyzer(BaseAnalyzer):
    """
    A class to compute the correlation matrix for multivariate analysis.

    Attributes:
        _visualizer (Visualizer): Instance of Visualizer for plotting.

    Methods:
        analyze(**kwargs) -> pd.DataFrame:
            Computes the correlation matrix.

        plot(**kwargs) -> None:
            Plots the correlation matrix using a heatmap.

    Notes:
        This class requires a Visualizer instance provided through dependency injection.

    Raises:
        ValueError: If the provided data is not suitable for correlation analysis.

    """

    @inject
    def __init__(
        self,
        visualizer: Visualizer = Provide[VisualizeContainer.visualizer],
    ) -> None:
        """
        Initializes the CorrelationAnalyzer.

        Args:
            visualizer (Visualizer): Instance of Visualizer for plotting.
        """
        super().__init__()
        self._visualizer = visualizer

    def analyze(self, **kwargs) -> pd.DataFrame:
        """
        Computes the correlation matrix.

        Args:
            **kwargs: Additional arguments for computing correlation.

        Returns:
            pd.DataFrame: The correlation matrix.
        """
        return self._data.corr(**kwargs)

    def plot(self, **kwargs) -> None:
        """
        Plots the correlation matrix.

        Args:
            **kwargs: Additional arguments for plotting (optional).
        """
        corr_matrix = self.analyze()
        title = "Correlation Matrix"
        self._visualizer.heatmap(
            data=corr_matrix, annot=True, cmap="crest", title=title, **kwargs
        )
