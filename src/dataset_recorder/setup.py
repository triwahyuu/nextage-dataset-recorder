#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=["dataset_recorder"],
    package_dir={'dataset_recorder': 'dataset_recorder'},
    install_requires=[],
    # python_requires=">=3.8"
)
setup(**setup_args)
