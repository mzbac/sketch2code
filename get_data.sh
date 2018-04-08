#! /bin/bash
mkdir data
wget https://s3-us-west-2.amazonaws.com/sketch2code/data.zip -O data/all_data.zip
unzip data/all_data.zip -d data/all_data
