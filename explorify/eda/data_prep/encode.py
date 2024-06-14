#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Explorify                                                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /explorify/eda/data_prep/encode.py                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john@variancexplained.com                                                           #
# URL        : https://github.com/variancexplained/explorify                                       #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday June 13th 2024 08:05:57 pm                                                 #
# Modified   : Thursday June 13th 2024 08:26:48 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder

from explorify.eda.data_prep.base import BaseEncoder


# ------------------------------------------------------------------------------------------------ #
class EncoderFactory:
    """
    Factory class to create encoder objects.

    Methods:
        get_encoder(method: str, data: pd.DataFrame) -> BaseEncoder:
            Factory method to get the appropriate encoder.
    """

    @staticmethod
    def get_encoder(method: str, data: pd.DataFrame) -> BaseEncoder:
        """
        Factory method to get the appropriate encoder.

        Args:
            method (str): The method to use for encoding (e.g., 'onehot', 'label', 'target').
            data (pd.DataFrame): The dataset containing categorical variables to encode.

        Returns:
            BaseEncoder: The encoder object.
        """
        if method == "onehot":
            return OneHotEncoder(data)
        elif method == "label":
            return LabelEncoderHandler(data)
        elif method == "target":
            return TargetEncoderHandler(data)
        else:
            raise ValueError(f"Unknown method: {method}")


# ------------------------------------------------------------------------------------------------ #
class OneHotEncoder(BaseEncoder):
    """
    A class to handle one-hot encoding of categorical variables.

    Methods:
        encode(columns: list[str]) -> pd.DataFrame:
            Encodes the specified columns using one-hot encoding.
    """

    def encode(self, columns: list[str]) -> pd.DataFrame:
        """
        Encodes the specified columns using one-hot encoding.

        Args:
            columns (list[str]): The columns to apply one-hot encoding to.

        Returns:
            pd.DataFrame: The dataset with the specified columns one-hot encoded.
        """
        return pd.get_dummies(self._data, columns=columns)


# ------------------------------------------------------------------------------------------------ #
class LabelEncoderHandler(BaseEncoder):
    """
    A class to handle label encoding of categorical variables.

    Methods:
        encode(columns: list[str]) -> pd.DataFrame:
            Encodes the specified columns using label encoding.
    """

    def encode(self, columns: list[str]) -> pd.DataFrame:
        """
        Encodes the specified columns using label encoding.

        Args:
            columns (list[str]): The columns to apply label encoding to.

        Returns:
            pd.DataFrame: The dataset with the specified columns label encoded.
        """
        le = LabelEncoder()
        for column in columns:
            self._data[column] = le.fit_transform(self._data[column])
        return self._data


# ------------------------------------------------------------------------------------------------ #


class TargetEncoderHandler(BaseEncoder):
    """
    A class to handle target encoding of categorical variables.

    Methods:
        encode(columns: list[str], target: str) -> pd.DataFrame:
            Encodes the specified columns using target encoding.
    """

    def encode(self, columns: list[str], target: str) -> pd.DataFrame:
        """
        Encodes the specified columns using target encoding.

        Args:
            columns (list[str]): The columns to apply target encoding to.
            target (str): The target variable for calculating means.

        Returns:
            pd.DataFrame: The dataset with the specified columns target encoded.
        """
        te = TargetEncoder()
        self._data[columns] = te.fit_transform(self._data[columns], self._data[target])
        return self._data
