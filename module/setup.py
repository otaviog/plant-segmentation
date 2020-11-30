"""A setuptools based setup module for plant-segmentation
"""

import sys
import os

from setuptools import find_packages, setup


import torch
TORCH_ROOT = os.path.dirname(torch.__file__)


def _forbid_publish():
    argv = sys.argv
    blacklist = ['register', 'upload']

    for command in blacklist:
        if command in argv:
            values = {'command': command}
            print('Command "%(command)s" has been blacklisted, exiting...' %
                  values)
            sys.exit(2)


_forbid_publish()

REQUIREMENTS = [
    'segmentation-models-pytorch'
]


setup(
    name='plant-segmentation',
    version='0.0.1',
    description='Auxillary module for plant segmentation experiments',
    license='MIT',
    packages=find_packages(),
    install_requires=REQUIREMENTS,
)
