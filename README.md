# Adaptive Oscillator
Os can be described as a mathematical tool able to synchronize with a rhythmic and periodic signal by continuously estimating its fundamental features (i.e. frequency, amplitude, phase, and offset). For their properties, AOs found applications in gait pattern estimation strategies, where they are used to mimic the dynamics of the neuromechanical oscillators in charge of the rhythmical human locomotion. In addition, as gait periodicity can be captured by sensors recording joint kinematics, their application does not require a complex sensory network.

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
"""Basic usage for our module."""

def main() -> None:
    """Run a simple demonstration."""
    # Initialize system
    estimator = GaitPhaseEstimator(AOParameters())
    controller = LowLevelController()
    plotter = RealtimeAOPlotter() if args.real_time else None

    if plotter:
        logger.info("Starting Dash app for real-time plotting.")
        plotter.run()

    logger.info("Replaying log file with controller.")
    time_data, theta_il, theta_hat, phi_gp, omegas = run_controller_loop(
        log_data, estimator, controller, plotter=plotter
    )

if __name__ == "__main__":
    main()
```

## Results
The plot below shows the results being plotted in real time.
<img width="1491" height="1521" alt="Screenshot 2025-09-25 at 4 21 48 PM" src="https://github.com/user-attachments/assets/bd593195-ae56-4f48-97dd-728f58ae1dae" />

The plot below shows the results of plotting an entire saved dataset.
<img width="1500" height="800" alt="iit_plot" src="https://github.com/user-attachments/assets/25463eb0-8bcd-4935-8d53-d33c4e290a91" />
