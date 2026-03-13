# Computational Content Analysis Epistemology: The Domestication of Topic Modeling

## Overview
This repository contains the data-processing pipeline, analytical code, and manuscript materials for Domesticating the Algorithm: Epistemic Drift and the Erosion of Disciplinary Distinctiveness in Computational Text Analysis. The project examines how five social science disciplines—Sociology, Political Science, Communication, Psychology, and Interdisciplinary Social Sciences—adopt and stabilize unsupervised topic modeling in published research. Drawing on a corpus of 769 full-text articles published between 2009 and 2024, the repository documents the workflow for corpus construction, PDF parsing, section classification, large language model-based information extraction, variable construction, statistical analysis, and figure and table generation. It is designed as a reproducible research archive for tracing how documented implementation practices vary across disciplines and change over time.

## Methodology
The project pipeline involves several key stages:
* **Data Collection:** Harvesting academic corpus data from the Web of Science to track the usage of topic modeling in social science literature.
* **Data Extraction:** Utilizing Large Language Models (LLMs) to extract nuanced methodological details and implementation contexts from the collected papers.
* **Analysis:** Python-based analysis to map how the usage of these computational tools varies across disciplines over time.

## Repository Structure
The repository is organized into the following directories:

* `configs/`: Configuration files containing parameters for the data collection and LLM extraction pipelines.
* `outputs/`: Directory for storing generated results, processed datasets, and analytical outputs.
* `scripts/`: Python scripts (`.py` files and Jupyter/Colab notebooks) used for interacting with the Web of Science, running LLM extractions, and performing downstream data analysis.
* `tei/`: Stores Text Encoding Initiative (TEI) XML files or related text-encoded structures used in the processing of the academic literature.

## Prerequisites and Installation
This project relies entirely on Python. Recommended to run using a virtual environment or Google Colab.
