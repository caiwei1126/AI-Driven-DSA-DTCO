# AI-Driven Closed-Loop Simulation Framework for Directed Self-Assembly

This repository provides the code and supporting data associated with the manuscript **"AI-Driven Closed-Loop Simulation Framework for Directed Self-Assembly toward Design-Technology Co-Optimization"**.

The repository is organized to support review and reproduction of the analyses reported in the manuscript, including template-geometry generation, SEM data processing, CNN-based defect classification, Bayesian optimization analysis, inverse-design evaluation, and LLM-assisted assessment results.

## Repository Contents

```text
Code_2_0/
+-- Bayesian_Optimization/      # Bayesian optimization results and code scripts
+-- CNN/                        # CNN model definition, trained weights, and supporting dataset archive
+-- inverse_design/             # CCD response and inverse-design analysis scripts
+-- LLM/                        # LLM evaluation data and code scripts
+-- reference_OK_DSA/           # Reference materials related to DSA simulation and optimization
+-- sem_data/                   # SEM data examples and associated spreadsheets
+-- simulation inputs/          # Template-geometry generation and CCD extraction scripts
```

## Environment

Python 3.10 or newer is recommended. The required Python packages are listed in `requirements.txt` and can be installed with:

```bash
pip install -r requirements.txt
```

## Main Components

### Simulation Inputs

The `simulation inputs/` directory contains scripts for constructing template geometries and extracting CCD-related quantities. These scripts were used to prepare and analyze template configurations for DSA simulations.

### SEM Data

The `sem_data/` directory contains SEM data examples, processed masks, segmentation results, processed DSA data, and associated spreadsheets used in the analysis workflow.

### CNN-Based Defect Classification

The `CNN/` directory contains the CNN model architecture and trained model weights used for binary classification of DSA outcomes. The model definition is provided in `CNN/model.py`, and the trained weights are provided in `CNN/best_model_4_CNN_fixed.pth`.

The accompanying dataset archive contains defect and perfect DSA samples used for model development and evaluation.

### Bayesian Optimization

The `Bayesian_Optimization/` directory contains CSV data and scripts used to analyze Bayesian optimization performance, including acquisition values, convergence behavior, Gaussian-process results, parameter evolution, and prediction-error distributions.

### Inverse Design

The `inverse_design/` directory contains scripts and data for analyzing template-error results, CCD response with respect to design parameters, and selected parameter sets.

### LLM Evaluation

The `LLM/` directory contains evaluation inputs, evaluation labels, model-score tables, and code scripts used to summarize the LLM-assisted assessment results reported in the manuscript.

## Reproducing Analyses

Most analysis scripts are self-contained and read the corresponding local data files from their directories. They can be run directly with Python from the repository root or from the folder containing the script.

Scripts in `Bayesian_Optimization/`, `inverse_design/`, and `LLM/` support the corresponding analysis steps used during manuscript preparation.

## Citation

If this code or dataset is useful for your work, please cite the associated manuscript.
