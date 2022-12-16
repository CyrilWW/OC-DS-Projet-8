#! /bin/bash
set -x -e
sudo pip3 install pandas
sudo pip3 install Pillow
sudo pip3 install tensorflow
sudo pip3 install PyArrow
sudo pip3 install fsspec
sudo pip3 install s3fs

# Defaulting to user installation because normal site-packages is not writeable
# WARNING: The script markdown_py is installed in '/home/hadoop/.local/bin' which is not on PATH.
