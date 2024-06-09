#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Explorify                                                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /tests/test_bivariate/test_bivariate_categorical.py                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john@variancexplained.com                                                           #
# URL        : https://github.com/variancexplained/explorify                                       #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday June 9th 2024 11:54:12 am                                                    #
# Modified   : Sunday June 9th 2024 02:48:01 pm                                                    #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
import inspect
import logging
from datetime import datetime

import pandas as pd
import pytest

from explorify.eda.bivariate.categorical import (
    CategoricalBivariateAnalysis,
    NominalNominalBivariateAnalysis,
    NominalOrdinalBivariateAnalysis,
    OrdinalOrdinalBivariateAnalysis,
)
from explorify.eda.stats.inferential.base import StatTestResult

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=missing-class-docstring, line-too-long
# ------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.bivariate
@pytest.mark.category
class TestCategoricalBivariateAnalysis:  # pragma: no cover
    # ============================================================================================ #
    def test_validate_input(self, dataset, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        analysis = CategoricalBivariateAnalysis(data=dataset)
        with pytest.raises(ValueError):
            analysis.validate_input(var1="review_length", var2="category")
        with pytest.raises(ValueError):
            analysis.validate_input(var1="category", var2="review_length")
        with pytest.raises(ValueError):
            analysis.validate_input(var1="bogus", var2="category")
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_contigency_table(self, dataset, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        analysis = CategoricalBivariateAnalysis(data=dataset)
        df = analysis.contingency_table(var1="app_name", var2="category")
        assert isinstance(df, pd.DataFrame)
        logger.info(df.head())
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_chisquare(self, dataset, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        analysis = CategoricalBivariateAnalysis(data=dataset)
        result = analysis.chi_square(var1="app_name", var2="category")
        assert isinstance(result, StatTestResult)
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
    def test_cramersv(self, dataset, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        analysis = CategoricalBivariateAnalysis(data=dataset)
        result = analysis.cramers_v(var1="app_name", var2="category")
        assert isinstance(result, StatTestResult)
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
    def test_mutual_info(self, dataset, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        analysis = CategoricalBivariateAnalysis(data=dataset)
        result = analysis.mutual_information(var1="app_name", var2="category")
        assert isinstance(result, float)
        logger.info(result)
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)


# ------------------------------------------------------------------------------------------------ #
@pytest.mark.nominal
@pytest.mark.bivariate
@pytest.mark.category
class TestNominalNominalBivariateAnalysis:  # pragma: no cover
    # ============================================================================================ #
    def test_validate_input(self, credit, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        analysis = NominalNominalBivariateAnalysis(data=credit)
        with pytest.raises(ValueError):
            analysis.validate_input(var1="Income", var2="Gender")
        with pytest.raises(ValueError):
            analysis.validate_input(var1="Gender", var2="Income")
        with pytest.raises(ValueError):
            analysis.validate_input(var1="bogus", var2="Gender")
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_phi_coefficient(self, credit, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        analysis = NominalNominalBivariateAnalysis(data=credit)
        result = analysis.phi_coefficient(var1="Marital Status", var2="Gender")
        assert isinstance(result, float)
        logger.info(result)
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_phi_coefficient_not_2x2(self, credit, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        analysis = NominalNominalBivariateAnalysis(data=credit)
        with pytest.raises(ValueError):
            _ = analysis.phi_coefficient(var1="Education", var2="Gender")

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_contingency_coefficient(self, credit, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        analysis = NominalNominalBivariateAnalysis(data=credit)
        result = analysis.contingency_coefficient(var1="Marital Status", var2="Gender")
        assert isinstance(result, float)
        logger.info(result)
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_lambda_coefficient(self, credit, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        analysis = NominalNominalBivariateAnalysis(data=credit)
        result = analysis.lambda_coefficient(var1="Marital Status", var2="Gender")
        assert isinstance(result, float)
        logger.info(result)
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)


@pytest.mark.ordinal_nominal
@pytest.mark.bivariate
@pytest.mark.category
class TestNominalOrdinalBivariateAnalysis:  # pragma: no cover
    # ============================================================================================ #
    def test_validate_input(self, credit, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        analysis = NominalOrdinalBivariateAnalysis(data=credit)
        with pytest.raises(ValueError):
            analysis.validate_input(var1="Income", var2="Gender")
        with pytest.raises(ValueError):
            analysis.validate_input(var1="Gender", var2="Income")
        with pytest.raises(ValueError):
            analysis.validate_input(var1="bogus", var2="Gender")
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_gamma(self, credit, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        analysis = NominalOrdinalBivariateAnalysis(data=credit)
        result = analysis.gamma(var1="Gender", var2="Education")
        assert isinstance(result, float)
        logger.info(result)
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_lambda_coefficient(self, credit, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        analysis = NominalOrdinalBivariateAnalysis(data=credit)
        result = analysis.lambda_coefficient(var1="Gender", var2="Education")
        assert isinstance(result, float)
        logger.info(result)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_theil_u(self, credit, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        analysis = NominalOrdinalBivariateAnalysis(data=credit)
        result = analysis.theil_u(var1="Gender", var2="Education")
        assert isinstance(result, float)
        logger.info(result)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)


@pytest.mark.ordinal
@pytest.mark.bivariate
@pytest.mark.category
class TestOrdinalOrdinalBivariateAnalysis:  # pragma: no cover
    # ============================================================================================ #
    def test_validate_input(self, credit, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        analysis = OrdinalOrdinalBivariateAnalysis(data=credit)
        with pytest.raises(ValueError):
            analysis.validate_input(var1="Income", var2="Gender")
        with pytest.raises(ValueError):
            analysis.validate_input(var1="Gender", var2="Income")
        with pytest.raises(ValueError):
            analysis.validate_input(var1="bogus", var2="Gender")
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_cramersv(self, credit, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        analysis = OrdinalOrdinalBivariateAnalysis(data=credit)
        result = analysis.cramer_v(var1="Gender", var2="Education")
        assert isinstance(result, StatTestResult)
        logger.info(result)

    # ============================================================================================ #
    def test_gamma(self, credit, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        analysis = OrdinalOrdinalBivariateAnalysis(data=credit)
        result = analysis.gamma(var1="Gender", var2="Education")
        assert isinstance(result, float)
        logger.info(result)

    # ============================================================================================ #
    def test_kendalls_tau(self, credit, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        analysis = OrdinalOrdinalBivariateAnalysis(data=credit)
        result = analysis.kendalls_tau(var1="Gender", var2="Education")
        assert isinstance(result, StatTestResult)
        logger.info(result)

    # ============================================================================================ #
    def test_spearmans_rank(self, credit, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        analysis = OrdinalOrdinalBivariateAnalysis(data=credit)
        result = analysis.spearmans_rank(var1="Gender", var2="Education")
        assert isinstance(result, StatTestResult)
        logger.info(result)

    # ============================================================================================ #
    def test_mutual_information(self, credit, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        analysis = OrdinalOrdinalBivariateAnalysis(data=credit)
        result = analysis.mutual_information(var1="Gender", var2="Education")
        assert isinstance(result, float)
        logger.info(result)

    # ============================================================================================ #
    def test_contingency_table(self, credit, caplog):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        analysis = OrdinalOrdinalBivariateAnalysis(data=credit)
        result = analysis.contingency_table(var1="Gender", var2="Education")
        assert isinstance(result, pd.DataFrame)
        logger.info(result)
