#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation of biased coin tosses and inference using various evidence-based methods

Started on Tue Nov 12 17:06:46 2024
"""
"""  Run after generating data in simulations.py"""

import os
from matplotlib import pyplot as plt
import numpy as np
import pickle
import pandas as pd


def plot_mean_std_by_threshold_over_biasF(results, plot_partitioning, partitioning, thresholds, bias_lower, bias_upper, rule_labels, marker_styles, output_dir, n_tosses):
    """
    Plot mean/std results per threshold (x-axis), averaged over biases for each rule.
    """
    os.makedirs(output_dir, exist_ok=True)
    included_rules = ['IBE-Standard', 'IBE-StandardFiltered', 'IBE-ER']
    
    # Filter rule_labels and marker_styles accordingly
    filtered_rules = [(name, marker) for name, marker in zip(rule_labels, marker_styles) if name in included_rules]
    display_names = {
        'IBE-Standard': r'$\mathrm{IBE}_{\mathrm{Ba}}$',
        'IBE-StandardFiltered': r'$\mathrm{IBE}_{\mathrm{Fi}}$',
        'IBE-ER': r'$\mathrm{IBE}_{\mathrm{ER}}$'
    }
    for i, partition in enumerate(partitioning):
        current_partition_results = results[i]
        num_biases = partition
        if num_biases in plot_partitioning:
    
            # Collect mean/std across biases for each threshold and rule
            fig, ax = plt.subplots(figsize=(10, 6))
            # fig.suptitle(f'Partition: {partition} - Mean over Biases by Threshold', fontsize=16)
            position = -1
    
            for rule_idx, (rule_name, marker) in enumerate(filtered_rules):
                means = []
                stds = []
    
                for threshold_idx in range(len(thresholds)):
                    # Get all values for this threshold across biases
                    values = [bias_result[0][threshold_idx+1][rule_labels.index(rule_name)] for bias_result in current_partition_results]
                    means.append(np.mean(values))
                    stds.append(np.std(values, ddof=1))
    
                thresholds_jittered = [t + 0.01 * position for t in thresholds]
                ax.errorbar(thresholds_jittered, means, yerr=stds, label=display_names[rule_name], marker=marker,
                            linestyle='None', markersize=8, capsize=5)
                position += 0.5
            ax.set_xlabel('Certainty Threshold')
            ax.set_ylabel('Average Score ± Std (over simulated biases)')
            ax.set_xticks(thresholds)
            ax.axhline(0, color='lightgray', linestyle='--', linewidth=0.7)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            filename = os.path.join(output_dir, "fig2.pdf")
            plt.savefig(filename, format='pdf')
            plt.show()

def plot_all_rules_side_by_side(results_by_tosses, rule_labels, thresholds, output_dir):
    """
    Plot all 3 rules (IBE_Ba, IBE_Fi, IBE_ER) in one figure with subplots showing
    how each rule performs as toss count increases.
    """
    os.makedirs(output_dir, exist_ok=True)

    display_names = {
        'IBE-Standard': r'$\mathrm{IBE}_{\mathrm{Ba}}$',
        'IBE-StandardFiltered': r'$\mathrm{IBE}_{\mathrm{Fi}}$',
        'IBE-ER': r'$\mathrm{IBE}_{\mathrm{ER}}$'
    }

    included_rules = ['IBE-Standard', 'IBE-StandardFiltered', 'IBE-ER']
    toss_counts = sorted(results_by_tosses.keys())

    toss_colors = {
        10: '#E69F00',  # orange
        25: '#56B4E9',  # sky blue
        50: '#009E73',  # bluish green
        100: '#D55E00'  # vermillion
    }

    toss_markers = {
        10: 'o',   # circle
        25: 's',   # square
        50: '^',   # triangle
        100: 'D'   # diamond
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()  # Flatten to 1D list for easier looping

    for i, rule in enumerate(included_rules):
        ax = axes[i]
        rule_idx = rule_labels.index(rule)

        for j, n_tosses in enumerate(toss_counts):
            results = results_by_tosses[n_tosses]
            partition_result = results[1]

            means = []
            stds = []

            for t_idx in range(len(thresholds)):
                values = [bias_result[0][t_idx+1][rule_idx] for bias_result in partition_result]
                means.append(np.mean(values))
                stds.append(np.std(values, ddof=1))

            thresholds_jittered = [t + 0.01 * j for t in thresholds]
            if i==0:
                labelcustom=f'{n_tosses} tosses'
            else:
                labelcustom='_nolegend_'
            ax.errorbar(thresholds_jittered, means, yerr=stds,
                        label=labelcustom, linestyle='None', color=toss_colors[n_tosses],marker=toss_markers[n_tosses],
                        capsize=4)

        ax.set_title(display_names[rule])
        ax.set_xlabel('Certainty Threshold')
        ax.axhline(0, color='gray', linestyle='None', linewidth=0.7)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xticks(thresholds)
        if i % 2 == 0:
            ax.set_ylabel('Average Score ± Std')

    # Hide the unused 4th subplot
    if len(included_rules) < len(axes):
        axes[-1].axis('off')

    fig.suptitle('Performance of IBE Variants by Toss Count', fontsize=14)
    fig.legend(loc='center', bbox_to_anchor=(0.75, 0.18),ncol=4)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'fig3.pdf'))
    plt.show()

def load_results_by_partition(n_tosses, partitions=[3, 7, 11], path='results_dir/results_n_tosses_25_weight_positive_1_weight_negative_1_uncert_lower_0.5_upper_1.0.pkl'):
    with open(path, 'rb') as f:
        all_partition_results = pickle.load(f)

    results_by_partitions = {
        partition: [partition_result]
        for partition, partition_result in zip(partitions, all_partition_results)
    }
    return results_by_partitions

def plot_all_rules_by_partition(results_by_partitions, rule_labels, thresholds, output_dir):
    """
    Plot all 3 rules (IBE_Ba, IBE_Fi, IBE_ER) in one figure with subplots showing
    how each rule performs as toss count increases.
    """
    os.makedirs(output_dir, exist_ok=True)

    display_names = {
        'IBE-Standard': r'$\mathrm{IBE}_{\mathrm{Ba}}$',
        'IBE-StandardFiltered': r'$\mathrm{IBE}_{\mathrm{Fi}}$',
        'IBE-ER': r'$\mathrm{IBE}_{\mathrm{ER}}$'
    }

    included_rules = ['IBE-Standard', 'IBE-StandardFiltered', 'IBE-ER']
    partition_sizes = sorted(results_by_partitions.keys())


    partition_colors = {
        3: '#E69F00',  # orange
        7: '#56B4E9',  # sky blue
        11: '#D55E00',  # bluish green
    }

    partition_markers = {
        3: 'o',   # circle
        7: 's',   # square
        11: '^',   # triangle
        # 100: 'D'   # diamond
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()  # Flatten to 1D list for easier looping

    for i, rule in enumerate(included_rules):
        ax = axes[i]
        rule_idx = rule_labels.index(rule)

        for j, n_tosses in enumerate(partition_sizes):
            results = results_by_partitions[n_tosses]
            partition_result = results[0]

            means = []
            stds = []

            for t_idx in range(len(thresholds)):
                values = [bias_result[0][t_idx+1][rule_idx] for bias_result in partition_result]
                means.append(np.mean(values))
                stds.append(np.std(values, ddof=1))

            thresholds_jittered = [t + 0.01 * j for t in thresholds]
            if i==0:
                labelcustom=f'{n_tosses} possible coins'
            else:
                labelcustom='_nolegend_'
            ax.errorbar(thresholds_jittered, means, yerr=stds,
                        label=labelcustom, linestyle='None', color=partition_colors[n_tosses],marker=partition_markers[n_tosses],
                        capsize=4)

        ax.set_title(display_names[rule])
        ax.set_xlabel('Certainty Threshold')
        ax.axhline(0, color='gray', linestyle='None', linewidth=0.7)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xticks(thresholds)
        if i % 2 == 0:
            ax.set_ylabel('Average Score ± Std')

    # Hide the unused 4th subplot
    if len(included_rules) < len(axes):
        axes[-1].axis('off')

    # fig.suptitle('Performance of IBE Variants by Number of Hypotheses', fontsize=14)
    fig.legend(loc='center', bbox_to_anchor=(0.75, 0.18),ncol=4)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'fig4.pdf'))
    plt.show()

def plot_ibe_grid_by_case(file_map, thresholds, output_dir):
    """
    file_map: dict mapping subplot labels to file paths, e.g.,
        {
            'Simple': 'path/to/simple.csv',
            'Middling-Easier': 'path/to/middling_easier.csv',
            'Middling-Harder': 'path/to/middling_harder.csv',
            'Hard': 'path/to/hard.csv'
        }
    thresholds: list of thresholds to plot on x-axis
    output_dir: folder where PDF will be saved
    """

    os.makedirs(output_dir, exist_ok=True)

    display_names = {
        'IBE-Standard': r'$\mathrm{IBE}_{\mathrm{Ba}}$',
        'IBE-StandardFiltered': r'$\mathrm{IBE}_{\mathrm{Fi}}$',
        'IBE-ER': r'$\mathrm{IBE}_{\mathrm{ER}}$'
    }

    included_rules = ['IBE-Standard', 'IBE-StandardFiltered', 'IBE-ER']
    markers = ['o', 's', 'D']  # One per rule

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    axes = axes.flatten()
    titles = list(file_map.keys())

    for i, (title, file_path) in enumerate(file_map.items()):
        df = pd.read_csv(file_path)
        df = df[df["Threshold"] > 0.5]

        ax = axes[i]
        ax.set_title(title)

        for r_idx, rule in enumerate(included_rules):
            mean_col = f"{rule} Mean"
            std_col = f"{rule} Std"
            df_grouped = df.groupby("Threshold")[[mean_col, std_col]].mean().reset_index()

            x = df_grouped["Threshold"] + 0.01 * r_idx
            y = df_grouped[mean_col]
            yerr = df_grouped[std_col]

            ax.errorbar(x, y, yerr=yerr,
                        label=display_names[rule],
                        marker=markers[r_idx], linestyle='None', capsize=4, markersize=6)

        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xticks(thresholds)
        ax.set_xlabel('Certainty Threshold')
        if i % 2 == 0:
            ax.set_ylabel('Average Score ± Std')

    # fig.suptitle('IBE Variant Performance by Case', fontsize=16)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'fig5.pdf'))
    plt.show()



def fig2():
    data='results_dir/results_n_tosses_10_weight_positive_1_weight_negative_1_uncert_lower_0.5_upper_1.0.pkl'
    with open(data, 'rb') as f:
        results = pickle.load(f)
    plot_mean_std_by_threshold_over_biasF(results, [7], [3,7,11], [0.6,0.7,0.8,0.9], 0, 1.0,
                                         ['IBE-ER', 'IBE-Standard', 'IBE-StandardFiltered', 'JC', 'IBE-Star'], ['o', 's', 'D', '^', '*'], 'agreggated/', 10)


def fig3():
    results_by_tosses = {}
    for toss in [10, 25, 50, 100]:
        with open(f'results_dir/results_n_tosses_{toss}_weight_positive_1_weight_negative_1_uncert_lower_0.5_upper_1.0.pkl', 'rb') as f:
            results_by_tosses[toss] = pickle.load(f)

    plot_all_rules_side_by_side(
        results_by_tosses,
        rule_labels=['IBE-ER', 'IBE-Standard', 'IBE-StandardFiltered', 'JC', 'IBE-Star'],
        thresholds=[0.6, 0.7, 0.8, 0.9],
        output_dir='agreggated/'
    )
    
def fig4(n_tosses=10):
    results_by_partitions = load_results_by_partition(n_tosses)


    plot_all_rules_by_partition(
        results_by_partitions=results_by_partitions,
        rule_labels=['IBE-ER', 'IBE-Standard', 'IBE-StandardFiltered', 'JC', 'IBE-Star'],
        thresholds=[0.6, 0.7, 0.8, 0.9],
        output_dir='agreggated/'
    )


def fig5():
    file_map = {
        'Low Risk/Easy Task (100 tosses, 3 hyp., reward: 10, penalty: 0)': 'results_dir/results_n_tosses_100_weight_positive_10_weight_negative_0_uncert_lower_0.5_upper_1.0.pkl',
        'High Risk/Hard Task (10 tosses, 11 hyp., reward: 0, penalty: 10)': 'results_dir/results_n_tosses_10_weight_positive_0_weight_negative_10_uncert_lower_0.5_upper_1.0.pkl'
    }
    
    thresholds = [0.6, 0.7, 0.8, 0.9]  # adjust as needed
    output_dir = 'agreggated/'
    
    plot_ibe_grid_by_case(file_map, thresholds, output_dir)
    
