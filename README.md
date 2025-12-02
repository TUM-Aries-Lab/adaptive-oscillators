# Adaptive Oscillator
[![Coverage Status](https://coveralls.io/repos/github/TUM-Aries-Lab/adaptive-oscillators/badge.svg?branch=main)](https://coveralls.io/github/TUM-Aries-Lab/template-python?branch=main)
![Docker Image CI](https://github.com/TUM-Aries-Lab/adaptive-oscillators/actions/workflows/ci.yml/badge.svg)

AOs can be described as a mathematical tool able to synchronize with a rhythmic and periodic signal by continuously estimating its fundamental features (i.e. frequency, amplitude, phase, and offset). For their properties, AOs found applications in gait pattern estimation strategies, where they are used to mimic the dynamics of the neuromechanical oscillators in charge of the rhythmical human locomotion. In addition, as gait periodicity can be captured by sensors recording joint kinematics, their application does not require a complex sensory network.

## Install
To install the library run: `pip install adaptive-oscillator`

## Development
0. Install [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)
1. `make init` to create the virtual environment and install dependencies
2. `make format` to format the code and check for errors
3. `make test` to run the test suite
4. `make clean` to delete the temporary files and directories
5. `poetry publish --build` to build and publish to https://pypi.org/project/adaptive-oscillator


## Usage
```
"""Basic usage for the adaptive oscillator."""

def main() -> None:
    """Run a simple demonstration."""
    # Initialize system
    controller = AOController(show_plots=True)

    while True:
        t, x_axis_angle, x_axis_vel = read_from_sensor()
        th = np.deg2rad(x_axis_angle)
        dth = np.deg2rad(x_axis_vel)
        controller.step(t=t, x=th, x_dot=dth)

    controller.plot_results()

if __name__ == "__main__":
    main()
```

## Results
The plot below shows the results being plotted in real time.

<img src="docs/results.png" width="1000">

## Repo Structure
The files and folders below are used for development purposes. This repo tree can be updated with `make tree`.
<!-- TREE-START -->
```
├── docs
│   ├── adaptive oscillators.pdf
│   └── results.png
├── src
│   └── adaptive_oscillator
│       ├── log_files
│       │   ├── __init__.py
│       │   ├── main.py
│       │   └── parser.py
│       ├── utils
│       │   ├── __init__.py
│       │   ├── log_utils.py
│       │   ├── plot_utils.py
│       │   └── time_utils.py
│       ├── __init__.py
│       ├── __main__.py
│       ├── base_classes.py
│       ├── controller.py
│       ├── definitions.py
│       └── oscillator.py
├── tests
│   ├── __init__.py
│   ├── adaptive_oscillator_test.py
│   ├── base_classes_test.py
│   ├── conftest.py
│   ├── controller_test.py
│   ├── parser_utils_test.py
│   ├── plot_utils_test.py
│   ├── time_utils_test.py
│   └── utils_for_test.py
├── .gitignore
├── .pre-commit-config.yaml
├── CONTRIBUTING.md
├── LICENSE
├── Makefile
├── README.md
├── poetry.lock
├── pyproject.toml
└── repo_tree.py
```
<!-- TREE-END -->
