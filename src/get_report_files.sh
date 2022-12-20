#! /bin/bash
aws s3 cp s3://oc-ds-p8-fruits-project/ ./report/ --recursive --exclude "*" --include "report_*"