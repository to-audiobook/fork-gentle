#!/bin/bash

set -Ee

git submodule init
git submodule update

./install_deps.sh
(cd ext && ./install_kaldi.sh)
./install_models.sh
cd ext && make depend && make
