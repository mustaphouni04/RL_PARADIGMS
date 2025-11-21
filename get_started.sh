#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh 

conda activate mlagents

export TORCHDYNAMO_DISABLE=1

uv add "gymnasium[atari,accept-rom-license]"
uv add ale-py
