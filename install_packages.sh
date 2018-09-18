#!/bin/bash
echo "Starting installing packages."
conda install --yes --file requirements.txt
while read requirement; do conda install --yes $requirement; done < requirements.txt
pip install https://software.ecmwf.int/wiki/download/attachments/56664858/ecmwf-api-client-python.tgz
$SHELL
