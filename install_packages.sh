#!/bin/bash
echo "Starting installing packages."
conda install --yes --file requirements.txt
while read requirement; do conda install --yes $requirement; done < requirements.txt
pip install cdsapi
$SHELL
