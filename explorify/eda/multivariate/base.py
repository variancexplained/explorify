#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Explorify                                                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /explorify/eda/multivariate/base.py                                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john@variancexplained.com                                                           #
# URL        : https://github.com/variancexplained/explorify                                       #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday June 13th 2024 08:31:29 pm                                                 #
# Modified   : Thursday June 13th 2024 08:31:38 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod

import pandas as pd


# ------------------------------------------------------------------------------------------------ #
class BaseAnalyzer(ABC):
    """
    Abstract base class for multivariate analysis.

    Attributes:
        data (pd.DataFrame): The dataset to be analyzed.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initializes the BaseAnalyzer with the given dataset.

        Args:
            data (pd.DataFrame): The dataset to be analyzed.
        """
        self._data = data

    @abstractmethod
    def analyze(self, *args, **kwargs) -> pd.DataFrame:
        """
        Abstract method to perform the analysis. Must be implemented by subclasses.

        Returns:
            pd.DataFrame: The result of the analysis.
        """
        pass
