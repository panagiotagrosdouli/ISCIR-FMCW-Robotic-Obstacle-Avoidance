# ISCIR: FMCW Robotic Obstacle Avoidance

An open-source research project for studying how Frequency-Modulated Continuous-Wave (FMCW) sensing can support obstacle detection and reactive navigation for autonomous mobile robots.

## Research question

Can noisy FMCW range and velocity measurements be converted into safe, real-time motion decisions for a mobile robot?

## Initial scope

The first version focuses on a small and reproducible simulation pipeline:

```text
Synthetic scene
    -> FMCW-like measurements
    -> obstacle detection
    -> reactive controller
    -> robot motion
    -> evaluation metrics
```

The project does not require an external dataset at this stage. Scenes and ground-truth data are generated synthetically.

## Planned metrics

- collision rate
- goal success rate
- minimum obstacle clearance
- trajectory length
- completion time
- range-estimation error under different SNR levels

## Repository layout

```text
src/iscir/
  simulation/     # robot, obstacles and scenes
  sensing/        # FMCW sensor models
  robotics/       # obstacle-avoidance controllers
examples/         # runnable demonstrations
tests/            # automated tests
docs/             # architecture and roadmap
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
python examples/corridor_demo.py
```

## Current status

Early research prototype. The current milestone is a minimal two-dimensional simulator with synthetic FMCW-like range measurements and a rule-based obstacle-avoidance controller.

## Roadmap

1. Two-dimensional scene and robot model
2. Synthetic FMCW measurement model
3. Reactive obstacle avoidance
4. Range-Doppler processing
5. Monte Carlo evaluation across SNR values
6. Dynamic-obstacle tracking
7. ROS 2 or Webots integration

## License

MIT License.
