# LaBNeWT: Lattice Boltzmann Numerical Wave Tank

![CI](https://github.com/oscarjtg/labnewt/actions/workflows/ci.yml/badge.svg)

A Python implementation of the Lattice Boltzmann Method to model fluid flow in a rectangular domain.

## Installation

Clone the repository into your Python `user-site` directory
and use `pip` or `conda` to install.

```
cd `python -m site --user-site`
git clone https://github.com/oscarjtg/labnewt.git
cd labnewt
pip install .
```

or, if you wish to install in editable mode, replace the last line above with

```
pip install -e .
```

## Project structure

The project is composed of the following folders:

- `examples`: example scripts modelling simple fluid flows using the LaBNeWT `Model` and `Simulation` classes.
- `labnewt`: the project source code.
- `tests`: unit and integration tests.
