#!/bin/bash
# Setup directory structure for new experiment
EXP_DIR=$(pwd)"/experiments/$1"
mkdir "$EXP_DIR"
mkdir "$EXP_DIR/checkpoints/"
mkdir "$EXP_DIR/log/"
