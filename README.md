# CNNFS

CNNFS implements neural networks from scratch in C++, following the example of the _Neural Networks from Scratch in Python_ [book](https://nnfs.io) and [repo](https://github.com/Sentdex/nnfs).

## Features

- Dense (fully-connected) layers
- Matrix/vector math
- CUDA support
- RNG with the uniform and Gaussian distributions
- Multi-class toy dataset generation (spiral)

## How to Build

This repo has so far only been tested on Windows using the MSVC compiler. First, clone the repository with

```bat
git clone https://github.com/LucasAPayne/cnnfs.git
```

To build and run the project, simply navigate to the project directory and run

```bat
build
run
```

If the project is run without the batch file, it should be run like the following:

```bat
start build/cnnfs.exe
```
