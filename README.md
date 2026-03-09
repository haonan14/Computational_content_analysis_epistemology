# Computational Content Analysis Epistemology: The Domestication of Topic Modeling

## Overview
This repository contains the scripts, data pipelines, and configuration files for a research project analyzing the "domestication" of computational content analysis. Specifically, this project investigates how topic modeling methodologies are adopted, adapted, and implemented across different social science disciplines. 

By framing the adoption of computational methods through the lens of the sociology of science and knowledge, this research explores how different disciplinary cultures shape the application of text-as-data methodologies.

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
