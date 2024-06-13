#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Explorify                                                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /tests/test_bivariate/test_bivariate_numeric.py                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john@variancexplained.com                                                           #
# URL        : https://github.com/variancexplained/explorify                                       #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday June 9th 2024 04:00:12 pm                                                    #
# Modified   : Thursday June 13th 2024 11:34:23 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
import inspect
import logging
import os
from datetime import datetime

import pytest

from explorify.eda.bivariate.numeric import NumericBivariateAnalysis
from explorify.eda.regression.simple import RegressionResult
from explorify.eda.stats.inferential.base import StatTestResult
from explorify.eda.stats.inferential.correlation import (
    PearsonCorrelationTestResult,
    SpearmanCorrelationResult,
)
from explorify.eda.stats.inferential.gof import KSTestResult

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=missing-class-docstring, line-too-long
# ------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.bivariate
@pytest.mark.numeric
class TestNumericBivariateAnalysis:  # pragma: no cover
    # ============================================================================================ #
    def test_validate_input(self, reviews, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        analysis = NumericBivariateAnalysis(data=reviews)
        with pytest.raises(ValueError):
            analysis.validate_input(a_name="app_name", b_name="rating")
        with pytest.raises(ValueError):
            analysis.validate_input(a_name="rating", b_name="category")
        with pytest.raises(ValueError):
            analysis.validate_input(a_name="bogus", b_name="rating")
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_ttest(self, credit, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        a_name = "Male"
        a_data = credit.loc[credit["Gender"] == "Male"]["Income"].values
        b_name = "Female"
        b_data = credit.loc[credit["Gender"] == "Female"]["Income"].values
        varname = "Income"
        analysis = NumericBivariateAnalysis(data=credit)
        results = analysis.students_t_test(
            a_name=a_name, a_data=a_data, b_name=b_name, b_data=b_data, varname=varname
        )
        assert isinstance(results, StatTestResult)
        logger.info(results)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_kolmogorov_smirnov_test(self, reviews, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        analysis = NumericBivariateAnalysis(data=reviews)
        results = analysis.kolmogorov_smirnov_test(
            a_name="review_length", b_name="vote_sum"
        )
        assert isinstance(results, KSTestResult)
        assert isinstance(results.report, str)
        logger.info(results)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_kolmogorov_smirnov_test_bad_arguments(self, reviews, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        analysis = NumericBivariateAnalysis(data=reviews)
        with pytest.raises(ValueError):
            _ = analysis.kolmogorov_smirnov_test(a_name="bogus1", b_name="bogus2")

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_correlation(self, reviews, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        analysis = NumericBivariateAnalysis(data=reviews)
        result = analysis.correlation_coefficient(
            a_name="rating",
            b_name="review_length",
            normal=True,
        )
        assert isinstance(result, PearsonCorrelationTestResult)
        assert isinstance(result.report, str)
        logger.info(result)
        logger.info(result.report)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_correlation_non_normal(self, reviews, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        analysis = NumericBivariateAnalysis(data=reviews)
        result = analysis.correlation_coefficient(
            a_name="rating",
            b_name="review_length",
            normal=False,
        )
        assert isinstance(result, SpearmanCorrelationResult)
        assert isinstance(result.report, str)
        logger.info(result)
        logger.info(result.report)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_regression(self, reviews, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        analysis = NumericBivariateAnalysis(data=reviews)
        result = analysis.regression_analysis(a_name="review_length", b_name="rating")
        assert isinstance(result, RegressionResult)
        assert isinstance(result.report, str)
        logger.info(result.report)
        logger.info(result)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_image(self, plt, reviews, caplog):
        caplog.set_level(logging.INFO)
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        analysis = NumericBivariateAnalysis(data=reviews)
        os.makedirs("tests/plots/bivariate", exist_ok=True)
        plt.saveas = "tests/plots/bivariate/numeric_scatterplot.png"
        _ = analysis.visualize(a_name="review_length", b_name="rating")
        plt.show()

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)
