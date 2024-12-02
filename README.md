# CNNFS

CNNFS implements neural networks from scratch in C++, following the example of the _Neural Networks from Scratch in Python_ [book](https://nnfs.io) and [repo](https://github.com/Sentdex/nnfs). Currently, the goal of this repo is to reach feature parity with the book and repo, and work may continue beyond that.

## Features

- Dense (fully-connected) layers
- Matrix/vector math
- CUDA support
- RNG with the uniform and Gaussian distributions
- Multiclass toy dataset generation (spiral)

## How to Build

This repo has so far only been tested on Windows using the MSVC compiler. First, clone the repository with

    git clone https://github.com/LucasAPayne/cnnfs.git

To build and run the project, simply navigate to the project directory and run

    build
    run

If the project is run without the batch file, it should be run like the following:

    start ../build/cnnfs.exe
