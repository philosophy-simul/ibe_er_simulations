Inference to the Best Explanation from Uncertain Evidence: Simulations

This repository contains the simulation code used to evaluate various inference rules under uncertainty. The framework models agents attempting to infer the bias of a coin from sequences of noisy evidence (biased coin tosses), scored according to the accuracy and informativeness of their inferences.

Overview

The simulation compares multiple inference strategies, including:

+ IBE-ER (Evidentially Robust Inference)
* IBE-Standard (treats all evidence as certain)
* IBE-Filtered (disregards low-certainty evidence)

(Optional benchmarking rules such as Jeffrey Conditionalisation variants are mentioned in the paper and also included here.)

Running the Simulations

To run the simulations:

1. Ensure you have Python 3.x installed, as well as the necessary libraries (numpy, multiprocessing, etc; see functions.py).
2. Load both simulations.py and functions.py in your working directory.
3. Execute the simulations by running simulations.py.

The script supports multi-processing and will automatically parallelise over available cores.

Approximate runtime

With default settings:

Runtime: ~45 minutes

Hardware used: Apple M3 Pro, 18 GB RAM

Runtime varies depending on the number of repetitions, coin tosses, hypothesis partitions, and uncertainty thresholds. For faster testing, reduce the number of repetitions in simulations.py.

Outputs

The script generates:

1. Performance plots (PDFs) showing average score ± standard deviation for each rule across conditions.
2. CSV files with detailed results for further analysis

Citation / Use

If you use this code or adapt its structure for related projects, please cite the associated paper.
