<div align="center">

# Generalized Aberrations for Processing-Aware Optical Design

*ACM Transactions on Graphics (SIGGRAPH 2026)*

Geoffroi Côté &nbsp;&bull;&nbsp; Ethan Tseng &nbsp;&bull;&nbsp; Felix Heide

[**Visit the Project Page**](https://light.princeton.edu/generalized-aberrations)

</div>

This is the official code repository for the paper **"Generalized Aberrations for Processing-Aware Optical Design"**.

This repository provides the **EISOPTX** (End-to-End Imaging System Optimization of Imaging Optics) framework. It
includes tools for the end-to-end optimization of optical designs, supporting experiments ranging from standard lens
optimization to end-to-end setups that incorporate image restoration models.

---

## Table of Contents

- [Project Overview: EISOPTX](#project-overview-eisoptx)
- [Installation](#installation)
- [Configuration Files](#configuration-files)
    - [Standard Lens Optimization](#standard-lens-optimization)
    - [Image-Driven Optimization](#image-driven-optimization)
    - [End-to-End Optimization](#end-to-end-optimization)
- [Core Workflow](#core-workflow)
    - [Results and Visualization](#results-and-visualization)
    - [Example](#example)
- [Reproducing Manuscript Experiments](#reproducing-manuscript-experiments)
- [Citation](#citation)

---

## Project Overview: EISOPTX

The **EISOPTX** (End-to-End Imaging System Optimization of Imaging Optics) framework can define, optimize, and evaluate
complex imaging systems. It supports configurations for standard lens optimization, end-to-end setups with an image
restoration model, and more.

## Installation

To set up the environment, create a new conda environment and install dependencies using the provided `environment.yml`
file. For more PyTorch installation options, see [here](https://pytorch.org/get-started/locally/).

```bash
conda env create -n eisoptx -f environment.yml
conda activate eisoptx
```

## Configuration Files

Configuration files in `.yml` format specify parameters for experiments, covering model settings, optimization types,
and data. Each `.yml` file defines a complete experiment setup, including model and training configurations.

To see all configurable parameters and their descriptions, use the `--help` flag. This command will display a help
message for the `fit` command, which lists all options available for configuration:

```bash
python -m eisoptx.main fit --help
```

### Standard Lens Optimization

For standard lens optimization, required parameters include:

- `model.lens_parameterization`: Defines the lens sequence, initial parameters, and parameterization options like
  freezing variables.
- `model.ray_initialization`: Configures ray initialization based on optical system specifications like aperture and
  field.
- `model.residuals`: Specifies residuals for the least-squares objective, including weights and specific parameters.
- `model.lens_optimizer`: Specifies the optimizer configuration to use when optimizing the lens; this works with default
  PyTorch optimizers as well as our Levenberg-Marquardt algorithm implementation (LMOptimizer).

### Image-Driven Optimization

In addition to the standard fields, image-driven optimization setups include:

- `model.end_to_end_vector_mode`: Whether to use Generalized Transverse Ray Aberrations (`True`), fall back to the
  scalar loss (`False`), or disable the end-to-end mode entirely (`None`).
- `model.end_to_end_loss_weight`: Weight on the scalar end-to-end loss; if this setting is set to `None` and the
  end-to-end mode is enabled, the end-to-end loss will be monitored but not optimized.
- `model.optics_simulator`: Configures the imaging simulator to model realistic aberrations on input images.

### End-to-End Optimization

For the co-design of an image restoration model, additional fields include:

- `model.image_restoration_model`: Defines the image-to-image restoration model and parameters.
- `model.irm_optimizer`: Optimizer settings for training the image restoration model.
- `model.irm_lr_scheduler`: Scheduler configuration for the image restoration model.

## Core Workflow

Run standard and custom experiments with the following commands:

- **Fit**: Optimizes the imaging system or a subset of its components (lens/restoration model), based on parameters
  defined in the `.yml` configuration file, saving logs and checkpoints.
  ```bash
  python -m eisoptx.main fit -c [CONFIG].yml
  ```
- **Validate**: Evaluates the imaging system on a validation set, saving logs and figures.
  ```bash
  python -m eisoptx.main validate -c [CONFIG].yml
  ```
- **Test**: Evaluates only the lens without a validation set, saving logs and figures.
  ```bash
  python -m eisoptx.main test -c [CONFIG].yml
  ```

To override specific parameters, specify multiple `.yml` files:

```bash
python -m eisoptx.main fit -c [GENERAL].yml -c [SPECIFICS].yml
```

This applies parameters from `[GENERAL].yml` first, followed by any overrides in `[SPECIFICS].yml`.

### Results and Visualization

To visualize experiment results, launch Tensorboard:

```bash
tensorboard --logdir=logs
```

Access metrics and intermediate results under the **SCALARS** and **IMAGES** tabs. When Tensorboard logging is enabled
in the config, metrics and figures will be periodically saved during optimization, such as lens layouts or PSFs.

Each `fit`, `validate`, or `test` command generates a new subdirectory in `logs/`, organized by lens sequence. These
subdirectories store:

- The config file used for the run
- Model checkpoints
- Tensorboard logs
- Optional figures or outputs specified in the config

### Example

For instance, to evaluate a 4-element telephoto lens using the `demo_tele4p.yml` configuration, run the following
command:

```bash
python -m eisoptx.main test -c configs/demo_tele4p.yml
```

This will create a new directory in `logs/s-aRa-aRa-aRa-aRa-/version_0` with the following contents:

- `layout/00000.png`: Lens layout at step 0.
- `glass/00000.png`: Optimized and catalog glasses at step 0.
- `psfs/00000.png`: PSF for different fields and wavelengths at step 0.
- `lens_parameters/00000.yml`: Lens parameters at step 0 (can be copied to a new config).
- `config.yaml`: Copy of the configuration file.
- `events.out.tfevents.*`: Tensorboard logs.
- `lens.seq`: Code V sequence file.

> **Note**: The directory name `s-aRa-aRa-aRa-aRa-` represents the lens sequence notation, where `s` is the aperture
> stop, `R` is a refractive element, `aRa` is an aspherical refractive element, and `-` is an air gap.

## Reproducing Manuscript Experiments

To reproduce the experiments from the manuscript, use the provided configuration files in the `configs/` directory.

- **Toy Example Lenses**:
    - Test the optimized designs (replace `optimized_e2e.yml` with the desired design file):
      ```bash
      python -m eisoptx.main test -c configs/toy/defaults.yml -c configs/toy/designs/optimized_e2e.yml
      ```
    - Conventional optimization:
      ```bash
      python -m eisoptx.main fit -c configs/toy/defaults.yml -c configs/toy/[OPTIMIZER_FILE].yml
      ```
    - End-to-end optimization:
      ```bash
      python -m eisoptx.main fit -c configs/toy/defaults.yml -c configs/toy/defaults_e2e.yml -c configs/toy/[OPTIMIZER_FILE].yml
      ```

- **Smartphone Telephoto Lenses**:
    - Evaluate the optimized designs (replace `5p_tr80_spot.yml` with the desired design file):
      ```bash
      python -m eisoptx.main test -c configs/telephoto/defaults.yml -c configs/telephoto/designs/5p_tr80_spot.yml
      ```
    - Conventional optimization:
      ```bash
      python -m eisoptx.main fit -c configs/telephoto/defaults.yml -c [DESIGN_FILE].yml 
      ```
    - End-to-end optimization (the [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) must be installed under
      `data/` first):
      ```bash
      python -m eisoptx.main fit -c configs/telephoto/defaults.yml -c configs/telephoto/defaults_e2e.yml -c [DESIGN_FILE].yml
      ```

- **Microscope Objective Lenses**:
  Use the same commands as above, but replace `telephoto` with `microscope` in the paths.

- **Smartphone Wide-Angle Lenses**:
    - Replicate the optimization process from a rough starting point:
      ```bash
      python -m eisoptx.main fit -c configs/wide_angle/defaults.yml -c configs/wide_angle/designs/starting_point.yml
      ```
    - Evaluate the optimized design:
      ```bash
      python -m eisoptx.main test -c configs/wide_angle/defaults.yml -c configs/wide_angle/designs/ours.yml
      ```

- **C-Mount Lenses**:
    - Evaluate the optimized designs (replace `4p_spot.yml` with the desired design file):
      ```bash
      python -m eisoptx.main test -c configs/c_mount/defaults.yml -c configs/c_mount/designs/4p_spot.yml
      ```
    - End-to-end optimization for pre-restoration image quality:
      ```bash
      python -m eisoptx.main fit -c configs/c_mount/defaults.yml -c configs/c_mount/defaults_e2e_raw.yml -c configs/c_mount/designs/4p_spot.yml
      ```
    - End-to-end optimization for post-restoration image quality:
      ```bash
      python -m eisoptx.main fit -c configs/c_mount/defaults.yml -c configs/c_mount/defaults_e2e_restored.yml -c configs/c_mount/designs/4p_spot.yml
      ```

---

## Citation

If you find our work useful in your research, please cite:

```bibtex
@article{cote2026generalized,
  author    = {C\^{o}t'{e}, Geoffroi and Tseng, Ethan and Heide, Felix},
  title     = {Generalized Aberrations for Processing-Aware Optical Design},
  journal   = {ACM Trans. Graph.},
  year      = {2026},
  volume    = {45},
  number    = {5},
  month     = jun,
  articleno = {44},
  doi       = {10.1145/3817055}
}
```
