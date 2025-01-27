#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation of biased coin tosses and inference using various evidence-based methods (part of ongoing work with Finnur Dellsén)

Started on Tue Nov 12 17:06:46 2024
@author: boruttrpin
"""

from ibe_er import run_simulations,plot_results,plot_results2,plot_mean_std,export_results_to_csv



# Main Code
if __name__ == "__main__":
    # Configuration
    n_repetitions=1000
    ntosses_list = [10, 25, 50, 100]
    partitioning = [3, 5, 7, 9, 11]
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bias_lower = 0.1
    bias_upper = 0.9
    weight_positive=1
    weight_negative=1
    rule_labels = [r'IBE$_{ER}$', r'IBE$_{St}$', r'Inf$_{JC}$', r'IBE$_{JC}$']
    marker_styles = ['o', 's', 'D', '^']  # Markers for each rule
    test=False
    prefix=""
    if test:
        prefix="test/"
        n_repetitions=10
    else:
        prefix="bonus_"+str(weight_positive)+"_malus_"+str(weight_negative)+"_nr_runs_"+str(n_repetitions)+"/"
    output_dir_plots = prefix+"plots-joint"
    output_dir_plots_single = prefix+"plots-single"
    output_dir_data=prefix+"data"

    # Run simulations and generate plots
    for ntosses in ntosses_list:
        print(f"Running simulations for n_tosses: {ntosses}")
        results = run_simulations(ntosses, n_repetitions, bias_lower, bias_upper, partitioning, thresholds,weight_positive,weight_negative)
        plot_results(results, partitioning, thresholds, bias_lower, bias_upper, rule_labels, marker_styles, output_dir_plots, ntosses)
        plot_results2(results, partitioning, thresholds, bias_lower, bias_upper, rule_labels, marker_styles, output_dir_plots, ntosses)
        plot_mean_std(results, partitioning, bias_lower, bias_upper, rule_labels, marker_styles, output_dir_plots_single, ntosses)
        export_results_to_csv(results, partitioning, thresholds, bias_lower, bias_upper, rule_labels, output_dir_data, ntosses)
