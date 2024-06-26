#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Explorify                                                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.3                                                                              #
# Filename   : /tests/conftest.py                                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john@variancexplained.com                                                           #
# URL        : https://github.com/variancexplained/explorify                                       #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday April 25th 2024 12:55:55 am                                                #
# Modified   : Wednesday June 26th 2024 11:33:47 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from explorify import DataClass
from explorify.container import VisualizeContainer
from explorify.utils.io import IOService

# ------------------------------------------------------------------------------------------------ #
collect_ignore_glob = ["tests/data/*.*"]
# ------------------------------------------------------------------------------------------------ #
# pylint: disable=redefined-outer-name, no-member
# ------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------ #
logging.getLogger("matplotlib").setLevel(logging.WARNING)
# ------------------------------------------------------------------------------------------------ #
CREDIT_FP = "tests/data/credit.csv"
CASES_FP = "tests/data/calc_cases.csv"
CATEGORY_FP = "tests/data/categories.csv"


# ------------------------------------------------------------------------------------------------ #
#                                   DATASET FIXTURE                                                #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=False)
def reviews():
    """Returns test data."""
    FP = "tests/data/dataset.pkl"
    df = IOService.read(filepath=FP)
    return df


# ------------------------------------------------------------------------------------------------ #
#                                         CREDIT DATA                                              #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=False)
def credit():
    df = pd.read_csv(CREDIT_FP, index_col=None)
    df = df.astype(
        {
            "Gender": "category",
            "Age": np.int64,
            "Income": np.int64,
            "Children": np.int64,
            "Marital Status": "object",
            "Credit Rating": "category",
            "Education": "category",
        }
    )
    return df


# ------------------------------------------------------------------------------------------------ #
#                                       CATEGORY DATA                                              #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=False)
def categories():
    df = pd.read_csv(CATEGORY_FP, index_col=None)
    df = df.astype(
        {
            "Category": "str",
            "Pct Active Apps": float,
            "Selected": str,
        }
    )
    return df


# ------------------------------------------------------------------------------------------------ #
#                                   MODEL DATA FIXTURE                                             #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=False)
def model_data(credit):
    """Returns encoded and scaled data."""
    data = pd.get_dummies(data=credit, dtype=int)
    enc = StandardScaler()
    a1 = enc.fit_transform(data)
    return pd.DataFrame(data=a1, columns=enc.get_feature_names_out())


# ------------------------------------------------------------------------------------------------ #
#                                      DATACLASS                                                   #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class TestDataClass(DataClass):
    name: str = "test"
    size: int = 8329
    length: float = 920932.98
    dt: datetime = datetime.now()


# ------------------------------------------------------------------------------------------------ #
#                              DEPENDENCY INJECTION                                                #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=True)
def container():
    container = VisualizeContainer()
    container.init_resources()
    container.wire(packages=["explorify"])

    return container
