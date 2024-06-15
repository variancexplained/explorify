#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Explorify                                                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /tests/test_data_prep/test_outlier_handlers.py                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john@variancexplained.com                                                           #
# URL        : https://github.com/variancexplained/explorify                                       #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday June 14th 2024 09:10:41 pm                                                   #
# Modified   : Saturday June 15th 2024 03:36:38 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
import inspect
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from explorify.eda.data_prep.outliers import (
    CustomThresholdOutlierHandler,
    IQROutlierHandler,
    OutlierHandlerFactory,
    ZScoreOutlierHandler,
)

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=missing-class-docstring, line-too-long
# ------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.data_prep
@pytest.mark.outliers
class TestOutlierHandlers:  # pragma: no cover
    # ============================================================================================ #
    def test_zscore_outlier_handler(self, reviews, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        oh = ZScoreOutlierHandler(data=reviews)
        mean1 = np.mean(reviews["review_length"])
        result = oh.remove_outliers(column="review_length")
        mean2 = np.mean(result["review_length"])
        assert isinstance(result, pd.DataFrame)
        assert mean1 != mean2
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_iqr_outlier_handler(self, reviews, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        oh = IQROutlierHandler(data=reviews)
        mean1 = np.mean(reviews["review_length"])
        result = oh.remove_outliers(column="review_length")
        mean2 = np.mean(result["review_length"])
        assert isinstance(result, pd.DataFrame)
        assert mean1 != mean2
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_custom_outlier_handler(self, credit, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        oh = CustomThresholdOutlierHandler(data=credit)
        cutoff = np.max(credit["Income"]) * 0.8
        size_b4 = credit.shape[0]
        mean_b4 = np.mean(credit["Income"])
        result = oh.remove_outliers(column="Income", upper_bound=cutoff)
        size_afta = result.shape[0]
        mean_afta = np.mean(result["Income"])
        assert isinstance(result, pd.DataFrame)
        assert mean_b4 != mean_afta
        assert size_b4 != size_afta
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_custom_outlier_handler_lower_bound(self, credit, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        oh = CustomThresholdOutlierHandler(data=credit)
        cutoff = np.max(credit["Income"]) * 0.8
        size_b4 = credit.shape[0]
        mean_b4 = np.mean(credit["Income"])
        result = oh.remove_outliers(column="Income", lower_bound=0, upper_bound=cutoff)
        size_afta = result.shape[0]
        mean_afta = np.mean(result["Income"])
        assert isinstance(result, pd.DataFrame)
        assert mean_b4 != mean_afta
        assert size_b4 != size_afta
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_outlier_handler_factory(self, credit, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        factory = OutlierHandlerFactory()
        oh = factory.get_handler(method="zscore", data=credit)
        assert isinstance(oh, ZScoreOutlierHandler)
        oh = factory.get_handler(method="iqr", data=credit)
        assert isinstance(oh, IQROutlierHandler)
        oh = factory.get_handler(method="custom", data=credit)
        assert isinstance(oh, CustomThresholdOutlierHandler)
        with pytest.raises(ValueError):
            _ = factory.get_handler(method="bogus", data=credit)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)
