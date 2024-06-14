#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Explorify                                                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /explorify/__main__.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john@variancexplained.com                                                           #
# URL        : https://github.com/variancexplained/explorify                                       #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday June 13th 2024 05:37:52 pm                                                 #
# Modified   : Thursday June 13th 2024 06:56:16 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
import sys

from dependency_injector.wiring import Provide, inject

from explorify.container import VisualizeContainer
from explorify.eda.visualize.visualizer import Visualizer
from explorify.utils.io import IOService


# ------------------------------------------------------------------------------------------------ #
def get_data():
    fp = "tests/data/dataset.pkl"
    data = IOService.read(fp)
    data = data.select_dtypes(include=["number"])
    return data.corr()


@inject
def main(visualizer: Visualizer = Provide[VisualizeContainer.visualizer]):

    visualizer = visualizer
    visualizer.heatmap(data=get_data())


if __name__ == "__main__":
    container = VisualizeContainer()
    container.init_resources()
    container.wire(modules=[sys.modules[__name__]], packages=["explorify.eda"])
    main()
