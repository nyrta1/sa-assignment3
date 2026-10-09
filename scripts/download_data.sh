#!/usr/bin/env bash
# Download the OULAD mirror from Kaggle and unzip it into data/raw/.
# Kaggle may require you to be logged in; if curl fails, download the zip manually from
# https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad
# and put it at data/student-demographics-online-education-dataoulad.zip
set -euo pipefail
cd "$(dirname "$0")/.."
ZIP=data/student-demographics-online-education-dataoulad.zip
mkdir -p data/raw
if [ ! -f "$ZIP" ]; then
  curl -L -o "$ZIP" \
    https://www.kaggle.com/api/v1/datasets/download/anlgrbz/student-demographics-online-education-dataoulad
fi
unzip -o "$ZIP" -d data/raw
echo "Files in data/raw:"
find data/raw -name "*.csv" -exec ls -lh {} \;
