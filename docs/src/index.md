![bluebird-logo](images/bluebird_logo.png)


## Project Overview

Project Bluebird is a £13.7m EPSRC Prosperity Partnership between NATS, The University of Exeter and The Alan Turing Institute to augment and optimise air-traffic control using multi-agent systems. Our ambition is to deliver the world’s first AI system to control a section of airspace in live shadow trials, working with Air Traffic Controllers to help manage the complexities of their role.

The project has three research themes:

- Building a Digital Twin of UK airspace.
- Developing AI agents that can perform Air Traffic Control (ATC) within this digital twin environment.
- Safety, trustworthiness and explainability of AI in safety-critical systems such as ATC.

<video id="introVideo" autoplay muted loop playsinline width="1000">
  <source src="assets/InfiniteSpringfield.mp4" type="video/mp4">
</video>
<script>
  document.getElementById('introVideo').playbackRate = 0.2;
</script>

## Package Documentation
The BluebirdATC repo is made up of four packages, `bluebird-dt`, `bluebird-api`, `bluebird-gymnasium`, and `bluebird-hmi`.  Click on the boxes in the section below to see the documentation for each one.

<div class="grid cards" markdown>

- :material-cube-outline: **bluebird-dt**
  Core digital twin package.
  [Getting started →](bluebird-dt/getting-started.md)

- :material-api: **bluebird-api**
  HTTP interface for the twin.
  [Getting started →](bluebird-api/getting-started.md)

- :material-robot: **bluebird-gymnasium**
  Gym wrapper for training agents.
  [Getting started →](bluebird-gymnasium/getting-started.md)

- :material-monitor: **bluebird-hmi**
  React UI for visualization.
  [Getting started →](bluebird-hmi/developer.md)

</div>

If you are unfamiliar with ATC concepts, an introduction is available [here](atc/index.md).

### Versioning

Development of the BluebirdATC Digital Twin is still in early stages, with new features and bug fixes causing breaking changes.
This is why the current versions are still '0.x.x', reflecting that each 'MINOR' version could potentially have breaking changes, following the [semantic versioning](https://semver.org/#spec-item-5) conventions.
On the other hand, 'PATCH' versions are used for bug fixes and non-breaking changes, although they may subtly change the behaviour of the models.

Release notes available on Github will include a list of changes, specifying if any are breaking changes.

## References

- **A Probabilistic Digital Twin of UK Airspace**, AIAA SciTech Forum (2026): [paper](https://doi.org/10.48550/arXiv.2601.03113)
- **A framework for assuring the accuracy and fidelity of an AI-enabled Digital Twin of en route UK airspace**, AIAA SciTech Forum (2026): [paper](https://doi.org/10.48550/arXiv.2601.03120)
- **Human-in-the-Loop Testing of AI Agents for Air Traffic Control with a Regulated Assessment Framework**, AIAA SciTech Forum (2026): [paper](https://doi.org/10.48550/arXiv.2601.04288)
- **Fast Surrogate Models for Adaptive Aircraft Trajectory Prediction in En route Airspace**, AIAA SciTech Forum (2026): [paper](https://doi.org/10.48550/arXiv.2601.03075)
- **Online Action-Stacking Improves Reinforcement Learning Performance for Air Traffic Control**, AIAA SciTech Forum (2026): [paper](https://doi.org/10.48550/arXiv.2601.04287)
- **Conditioning Aircraft Trajectory Prediction on Meteorological Data with a Physics-Informed Machine Learning Approach**, AIAA SciTech Forum (2026): [paper](https://doi.org/10.48550/arXiv.2601.03152)
- **A Future Capabilities Agent for Tactical Air Traffic Control**, AIAA SciTech Forum (2026): [paper](https://arxiv.org/abs/2601.04285)
- **Towards Transparent AI Agents for Air Traffic Control**, AIAA SciTech Forum (2026): [paper](http://dx.doi.org/10.2139/ssrn.6042354)
