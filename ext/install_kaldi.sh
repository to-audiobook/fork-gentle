#!/bin/bash

jobs="$(($(../npproc.sh) + 1))";

# without this Kaldi will do us a "favor" and change the default python
mkdir -p 'kaldi/tools/python';
touch 'kaldi/tools/python/.use_default_python'

# Prepare Kaldi
cd kaldi/tools

make clean
make -j"$jobs"
cd ../src
# make clean (sometimes helpful after upgrading upstream?)
./configure --shared --debug-level=0 --use-cuda=yes --cudatk-dir=/usr/local/cuda
make clean
make -j"$jobs" depend
make -j"$jobs" online2 nnet3 nnet2 decoder feat hmm matrix util
cd ../../
