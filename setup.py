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


    description="Tools for easy and efficient handling of Multiscale Eleptrophysiology Format (MEF3).",
    long_description="MefWriter and MefReader are a high-level API tools containing all headers required for convenient MEF3 file writing and reading including support for MEF3 annotations.",
    long_description_content_type="",

    packages=setuptools.find_packages(),

    classifiers=[
        "Development Status :: 4 - Beta",
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        'Topic :: Scientific/Engineering :: Medical Science Apps.'
    ],
    python_requires='>=3.6',
    install_requires =[
        'numpy',
        'pandas',
        'pymef'
    ]
)

