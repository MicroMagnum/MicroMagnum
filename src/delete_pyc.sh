#!/bin/sh
find . -name \*.pyc -delete
find . -name \*.pyo -delete

# To disable the generation of .pyc files, use
# > export PYTHONDONTWRITEBYTECODE=1
# This works since Python 2.6
