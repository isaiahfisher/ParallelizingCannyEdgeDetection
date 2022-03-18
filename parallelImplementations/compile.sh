#!/bin/bash
module load learning/conda-5.1.0-py36-cpu
module load ml-toolkit-cpu/opencv/3.4.3
mpiicpc -std=c++11 traditionalCannyAlgorithm.cpp -o ParallelCanny `pkg-config --libs opencv` `pkg-config --cflags opencv`
