# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import setuptools

from setuptools import Command, Extension
import shlex
import subprocess
import os
import re


## get version from file
VERSIONFILE="./mef_tools/__init__.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setuptools.setup(
    name="mef_tools",
    version=verstr,
    license='Apache',
    url="https://github.com/mselair/MEF_Tools",

    author="Filip Mivalt",
    author_email="mivalt.filip@mayo.edu",


    description="Tools for handling Multiscale Eleptrophysiology Format (MEF3) using pymef library.",
    long_description="MefWriter and MefReader are high-level API tools containing all headers required for convenient MEF3 file writing and reading.",
    long_description_content_type="",

    packages=setuptools.find_packages(),

    classifiers=[
        "Development Status :: 3 - Alpha",
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering :: Medical Science Apps.'
    ],
    python_requires='>=3.6',
    install_requires =[
        'numpy',
        'pandas',
        'scipy',
        'sqlalchemy',
        'pyzmq',
        'tqdm',
        'sshtunnel',
        'pymef',
        'python-dateutil'
    ]
)

