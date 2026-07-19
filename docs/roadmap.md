# Roadmap

## v0.1 — Robotics prototype

- [x] Package structure
- [x] Two-dimensional robot and obstacle models
- [x] Synthetic range and radial-velocity measurements
- [x] Rule-based reactive controller
- [x] Corridor demonstration
- [x] Initial tests
- [ ] Simulation metrics and experiment logging

## v0.2 — FMCW signal processing

- [ ] Linear FMCW chirp generation
- [ ] Delayed and Doppler-shifted target echoes
- [ ] Beat-signal generation
- [ ] Range FFT
- [ ] Range-Doppler map
- [ ] Detection thresholding or CA-CFAR

## v0.3 — Scientific evaluation

- [ ] Monte Carlo experiments across SNR values
- [ ] Range and velocity RMSE
- [ ] Collision and success rates
- [ ] Minimum-clearance statistics
- [ ] Reproducible plots and result tables

## v0.4 — Dynamic robotics

- [ ] Moving-obstacle tracking
- [ ] Time-to-collision estimation
- [ ] Improved local planning
- [ ] Webots or ROS 2 integration

## Design principle

The robotics interface should remain independent of the sensing implementation. The current measurement-level sensor will later be replaceable by the full FMCW signal-processing pipeline without rewriting the controller or simulation environment.
