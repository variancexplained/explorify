#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Explorify                                                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.10                                                                             #
# Filename   : /tests/test_stats/test_inferential/test_wilcoxon.py                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/variancexplained/explorify                                       #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday June 5th 2023 09:32:36 pm                                                    #
# Modified   : Thursday June 13th 2024 11:23:28 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import inspect
import logging
from datetime import datetime

import pytest

from explorify.eda.stats.inferential.base import StatTestProfile
from explorify.eda.stats.inferential.centrality import WilcoxonSignedRankTest

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.stats
@pytest.mark.centrality
@pytest.mark.wilcoxon
class TestWilcoxonSignedRankTest:  # pragma: no cover
    # ============================================================================================ #
    def test_wilcoxon_test(self, credit, caplog):
        start = datetime.now()
        logger.info(
            "\n\nStarted {} {} at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                start.strftime("%I:%M:%S %p"),
                start.strftime("%m/%d/%Y"),
            )
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        a = credit.loc[credit["Education"] == "Bachelor's Degree"]["Income"][:30]
        b = credit.loc[credit["Education"] == "Master's Degree"]["Income"][:30]
        test = WilcoxonSignedRankTest(
            a_data=a,
            a_name="Bachelor's Degree",
            b_data=b,
            b_name="Master's Degree",
            varname="Income",
            correction=True,
        )
        test.run()
        assert isinstance(test.result.H0, str)
        assert isinstance(test.result.value, float)
        assert isinstance(test.result.pvalue, float)
        assert test.result.alpha == 0.05
        assert isinstance(test.profile, StatTestProfile)
        logging.debug(test.result)
        logging.debug(test.result.report)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            "\nCompleted {} {} in {} seconds at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                duration,
                end.strftime("%I:%M:%S %p"),
                end.strftime("%m/%d/%Y"),
            )
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_wilcoxon_test_wo_correction(self, credit, caplog):
        start = datetime.now()
        logger.info(
            "\n\nStarted {} {} at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                start.strftime("%I:%M:%S %p"),
                start.strftime("%m/%d/%Y"),
            )
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        a = credit.loc[credit["Education"] == "Bachelor's Degree"]["Income"][:30]
        b = credit.loc[credit["Education"] == "Master's Degree"]["Income"][:30]
        test = WilcoxonSignedRankTest(
            a_data=a,
            a_name="Bachelor's Degree",
            b_data=b,
            b_name="Master's Degree",
            varname="Income",
            correction=False,
        )
        test.run()
        assert isinstance(test.result.H0, str)
        assert isinstance(test.result.value, float)
        assert isinstance(test.result.pvalue, float)
        assert test.result.alpha == 0.05
        assert isinstance(test.profile, StatTestProfile)
        logging.debug(test.result)
        logging.debug(test.result.report)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            "\nCompleted {} {} in {} seconds at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                duration,
                end.strftime("%I:%M:%S %p"),
                end.strftime("%m/%d/%Y"),
            )
        )
        logger.info(single_line)
