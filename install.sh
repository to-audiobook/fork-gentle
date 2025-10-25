#!/bin/bash

set -e


jobs="$(($(./npproc.sh) + 1))";

git submodule init
git submodule update --remote --merge

./install_deps.sh
(cd ext && ./install_kaldi.sh)
./install_models.sh
cd ext && make -j"$jobs" depend && make -j"$jobs"
