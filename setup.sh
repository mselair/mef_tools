#!/bin/bash

python3 -m pip install --user --upgrade setuptools wheel
python3 setup.py sdist bdist_wheel

PATH_TAR=$(find dist -type f -name \*.tar.gz)
echo $PATH_TAR
python3 -m pip install $PATH_TAR






