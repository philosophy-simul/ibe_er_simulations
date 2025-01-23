#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation of biased coin tosses and inference using various evidence-based methods (part of ongoing work with Finnur Dellsén)

Started on Tue Nov 12 17:06:46 2024
@author: boruttrpin
"""

import numpy as np
from itertools import chain, repeat, count, islice
from collections import Counter
from typing import List, Dict, Tuple, Optional, Union
import os
from matplotlib import pyplot as plt
import csv
import datetime

def coin_toss_sequence(bias: float, n: int) -> List[int]:
    """
    Generate a sequence of biased coin tosses.

    Parameters:
        bias (float): Probability of heads (1).
        n (int): Number of tosses.

    Returns:
        List[int]: Sequence of coin toss outcomes (1: heads, 0: tails).
    """
    return (np.random.uniform(0, 1, n) <= bias).astype(int).tolist()


def uncertain_sequence(
    sequence: List[int], lower_bound: float = 0.5, upper_bound: float = 1.0
) -> List[Dict[str, float]]:
    """
    Generate uncertain evidence for a sequence of coin tosses.

    Parameters:
        sequence (List[int]): Sequence of coin toss outcomes.
        lower_bound (float): Minimum uncertainty value.
        upper_bound (float): Maximum uncertainty value.

    Returns:
        List[Dict[str, float]]: Sequence of toss outcomes with evidential certainty.
    """
    uncertainties = np.random.uniform(lower_bound, upper_bound, len(sequence))
    return [{"toss": toss, "evidential certainty": certainty} for toss, certainty in zip(sequence, uncertainties)]

def certain_uncertain_evidence(uncertain_sequence, threshold):
    certain_evidence = []
    uncertain_evidence = []
    evidence_ibeST = []
    
    # Loop through each entry in the list and apply the flipping condition
    for entry in uncertain_sequence:
        toss = entry['toss']
        evidCertainty = entry['evidential certainty']
        
        if evidCertainty >= threshold:
            certain_evidence.append(toss)
            evidence_ibeST.append(toss)

        else:
            uncertain_evidence.append(toss)
            if np.random.uniform(0,1)<=evidCertainty:
                evidence_ibeST.append(toss)
            else:
                evidence_ibeST.append(1-toss)



    return [certain_evidence,uncertain_evidence],evidence_ibeST




def generate_subsets(a, b):
    # Set to hold unique subsets (as tuples)
    unique_subsets = set()
    # Generate subsets of b, including the empty subset
    for r in range(len(b) + 1):
        observed_subsets=[]
        for subset_b in unique_combinations(b, r):
            if subset_b not in observed_subsets:
                observed_subsets.append(subset_b)
                # For each subset of b, combine it with all elements from a
                subset = tuple(a) + tuple(subset_b)
                unique_subsets.add(subset)  # Add the tuple to the set (removes duplicates)
    
    # Convert the set back to a list and return
    return [list(subset) for subset in unique_subsets]


def best_expl(sequence: List[int], hyp_space: List[float]) -> np.ndarray:
    """
    Identify the best explanation for the given sequence.

    Parameters:
        sequence (List[int]): Evidence sequence.
        hyp_space (List[float]): Hypothesis space (possible biases).

    Returns:
        np.ndarray: Closest hypothesis to observed ratio.
    """
    ratio = sum(sequence) / len(sequence)
    diffs = np.abs(np.array(hyp_space) - ratio)
    return np.array(hyp_space)[diffs == diffs.min()]


def jc_ibe_star(uncertain_sequence, priorJC=None, hyp_space=None, bonus=0.1):
    if hyp_space is None:
        hyp_space = [1/6, 2/6, 3/6, 4/6, 5/6]
    if priorJC is None:
        priorJC = [1 / len(hyp_space)] * len(hyp_space)  # Uniform prior by default
    priorIBE = priorJC
    posteriorJC = []
    posteriorIBE = []
    distrsJC = [priorJC[:]]
    distrsIBE = [priorJC[:]]
    biases = np.array(hyp_space)
    observed_tosses = []

    for outcome in uncertain_sequence:
        toss = outcome["toss"]
        p_e_post = outcome["evidential certainty"]
        
        # Update prior distributions
        if len(distrsJC) > 1:
            priorJC = posteriorJC
            priorIBE = posteriorIBE
        
        # Handle evidence based on subjective certainty
        if p_e_post > 0.5:
            observed_tosses.append(toss)
        elif p_e_post < 0.5:
            observed_tosses.append(1 - toss)
        else:
            observed_tosses.append(toss if np.random.uniform(0, 1) < 0.5 else 1 - toss)
        
        # Determine likelihoods for the current toss
        likelihoods = hyp_space if toss == 1 else [1 - p for p in hyp_space]
        
        # JC Update
        p_h_and_e = np.array([p_h_prior * p_e_given_h for p_h_prior, p_e_given_h in zip(priorJC, likelihoods)])
        p_h_and_neg_e = np.array([p_h_prior * (1 - p_e_given_h) for p_h_prior, p_e_given_h in zip(priorJC, likelihoods)])
        p_e_prior = np.sum(p_h_and_e)
        p_h_given_e = p_h_and_e / p_e_prior
        p_h_given_neg_e = p_h_and_neg_e / (1 - p_e_prior)
        posteriorJC = p_e_post * p_h_given_e + (1 - p_e_post) * p_h_given_neg_e

        # IBE Update
        IBE = np.array(best_expl(observed_tosses, biases))
        IBEindex = np.where(np.isin(biases, IBE))[0]
        p_h_and_e_ibe = np.array([p_h_prior * p_e_given_h for p_h_prior, p_e_given_h in zip(priorIBE, likelihoods)])
        p_h_and_neg_e_ibe = np.array([p_h_prior * (1 - p_e_given_h) for p_h_prior, p_e_given_h in zip(priorIBE, likelihoods)])
        p_h_and_e_ibe[IBEindex] += bonus / len(IBEindex)
        p_h_and_neg_e_ibe[IBEindex] += bonus / len(IBEindex) # The same explanation(s) are best for tails and heads relative to the coin toss in question, e.g. the best explanation of 3 heads wrt heads is full bias towards H, and 0 tails wrt to tails is full bias towards H.
        p_e_prior_ibe = np.sum(p_h_and_e_ibe)
        p_neg_e_prior_ibe = np.sum(p_h_and_neg_e_ibe)
        p_h_given_e_ibe = p_h_and_e_ibe / p_e_prior_ibe
        p_h_given_neg_e_ibe = p_h_and_neg_e_ibe / p_neg_e_prior_ibe
        posteriorIBE = p_e_post * p_h_given_e_ibe + (1 - p_e_post) * p_h_given_neg_e_ibe

        # Append updated distributions
        distrsJC.append(posteriorJC)
        distrsIBE.append(posteriorIBE)

    return distrsJC, distrsIBE

def single_experiment(
    true_bias: float,
    nr_tosses: int,
    evid_certainty_threshold: float,
    hyp_space: List[float],
    lower_bound: float = 0.5,
    upper_bound: float = 1.0,
    bonus: float = 0.1,
) -> Tuple[np.ndarray, ...]:
    """
    Conduct a single inference experiment with uncertain evidence.

    Parameters:
        true_bias (float): Actual bias of the coin.
        nr_tosses (int): Number of tosses in the experiment.
        evid_certainty_threshold (float): Certainty threshold for evidence classification.
        hyp_space (List[float]): Hypothesis space of biases.
        lower_bound (float): Minimum uncertainty bound.
        upper_bound (float): Maximum uncertainty bound.
        bonus (float): Bonus for best explanations.

    Returns:
        Tuple: Results from different inference methods.
    """
    hyp_space=np.array(hyp_space)
    sequence = coin_toss_sequence(true_bias, nr_tosses)
    uncertain_evidence = uncertain_sequence(sequence, lower_bound, upper_bound)
    certain_uncertain_ev=certain_uncertain_evidence(uncertain_evidence,evid_certainty_threshold)
    # ibe ER
    certain_partition=certain_uncertain_ev[0][0]
    uncertain_partition = certain_uncertain_ev[0][1]
    if certain_partition==[]:
        ibe_er_result=np.array(hyp_space)
        # if all evidence is uncertain, everything may be inferred (i.e., no informational value)
    elif uncertain_partition==[]:
        ibe_er_result=np.array(best_expl(certain_partition,hyp_space))
        # if all evidence is certain, then all evidence needs to be explained.
    else:
        # else: we need to explain all subsets of uncertain evidence combined with certain evidence.
        evidentialSubsets=generate_subsets(certain_partition,uncertain_partition)
        expls_ER=[best_expl(i,hyp_space) for i in evidentialSubsets]
        flattened_expls = np.concatenate(expls_ER)
        ibe_er_result = np.unique(flattened_expls)
    
    # ibe Standard: above some evidential threshold, gets the correct evidence, else it depends on the chance
    ibe_standard = best_expl(certain_uncertain_ev[1],hyp_space)


    # JC-IBE-Star inference
    probabilistic = jc_ibe_star(uncertain_evidence,None,hyp_space,bonus)
    jc_result=hyp_space[np.where(probabilistic[0][-1]==probabilistic[0][-1].max())]
    ibe_star_result=hyp_space[np.where(probabilistic[1][-1]==probabilistic[1][-1].max())]

    return ibe_er_result, ibe_standard,jc_result, ibe_star_result

def simulation(
    nr_repetitions, true_bias, nr_tosses, evid_uncertainty_threshold, 
    weight_positive=1, weight_negative=1, hyp_space=[1/6, 2/6, 3/6, 4/6, 5/6], 
    lower_bound=0.5, upper_bound=1, printing=True
):
    ibe_er_score = 0
    ibe_standard_score = 0
    jc_score = 0
    ibe_star_score = 0

    for run in range(nr_repetitions):
        res = single_experiment(true_bias, nr_tosses, evid_uncertainty_threshold, hyp_space, lower_bound, upper_bound)

        if true_bias in res[0]:
            ibe_er_score += ((len(hyp_space) - len(res[0])) / (len(hyp_space) - 1)) * weight_positive
        else:
            ibe_er_score -= weight_negative

        if true_bias in res[1]:
            ibe_standard_score += ((len(hyp_space) - len(res[1])) / (len(hyp_space) - 1)) * weight_positive
        else:
            ibe_standard_score -= weight_negative

        if true_bias in res[2]:
            jc_score += ((len(hyp_space) - len(res[2])) / (len(hyp_space) - 1)) * weight_positive
        else:
            jc_score -= weight_negative

        if true_bias in res[3]:
            ibe_star_score += ((len(hyp_space) - len(res[3])) / (len(hyp_space) - 1)) * weight_positive
        else:
            ibe_star_score -= weight_negative

    if printing:
        print("IBE-ER: " + str(ibe_er_score))
        print("IBE Standard: " + str(ibe_standard_score))
        print("JC: " + str(jc_score))
        print("IBE Star: " + str(ibe_star_score))

    return ibe_er_score, ibe_standard_score, jc_score, ibe_star_score


# Additional Helper Functions (Unchanged)
def unique_combinations(iterable, r):
    """Helper to generate unique combinations of an iterable."""
    values, counts = zip(*Counter(iterable).items())
    return unique_combinations_from_value_counts(values, counts, r)


def unique_combinations_from_value_counts(values, counts, r):
    """Helper for unique_combinations."""
    indices = list(islice(repeat_chain(count(), counts), r))
    if len(indices) < r:
        return
    while True:
        yield tuple(values[i] for i in indices)
        for i, j in zip(reversed(range(r)), repeat_chain(reversed(range(len(counts))), reversed(counts))):
            if indices[i] != j:
                break
        else:
            return
        j = indices[i] + 1
        for i, j in zip(range(i, r), repeat_chain(count(j), counts[j:])):
            indices[i] = j


def repeat_chain(values, counts):
    """Helper to repeat values according to counts."""
    return chain.from_iterable(map(repeat, values, counts))



def run_simulations(n_tosses,n_repetitions=1000, bias_lower=.1, bias_upper=.9, partitioning=[3,5,7,9,11], thresholds=[0.5,0.6,.7,.8,.9,1],weight_positive=1,weight_negative=1):
    """
    Run simulations for different partitions, thresholds, and biases.
    """
    results = []
    for partition in partitioning:
        print(f"Partition: {partition}")
        partition_results = []
        for threshold in thresholds:
            print(f"Threshold: {threshold}")
            threshold_results = []
            for bias in np.linspace(bias_lower, bias_upper, partition):
                print(f"Bias: {bias}")
                threshold_results.append(
                    simulation(n_repetitions, bias, n_tosses, threshold, weight_positive, weight_negative, 
                               np.linspace(bias_lower, bias_upper, partition), 0.5, 1, False)
                )
            partition_results.append(threshold_results)
        results.append(partition_results)
    return results

def plot_results(results, partitioning, thresholds, bias_lower, bias_upper, rule_labels, output_dir, n_tosses):
    """
    Generate and save plots for simulation results.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, partition in enumerate(partitioning):
        current_partition_results = results[i]
        num_biases = partition
        bias_values = np.linspace(bias_lower, bias_upper, num_biases)

        # Plot results for each threshold
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharey=True)
        fig.suptitle(f'Partition: {partition}', fontsize=16)

        for j, threshold_results in enumerate(current_partition_results):
            row, col = divmod(j, 3)
            ax = axes[row, col]

            for rule_idx, rule_name in enumerate(rule_labels):
                rule_values = [bias_result[rule_idx] for bias_result in threshold_results]
                ax.plot(bias_values, rule_values, label=rule_name)

            ax.set_title(f'Threshold: {thresholds[j]:.1f}')
            ax.set_xlabel('Bias')
            ax.grid(True)
            if col == 0:
                ax.set_ylabel('Values')

        # Hide unused subplots
        for j in range(len(thresholds), 6):
            row, col = divmod(j, 3)
            axes[row, col].axis('off')

        # Add legend to the last subplot
        axes[-1, -1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        filename = os.path.join(output_dir, f"n_tosses_{n_tosses}_partition_{partition}_plot.pdf")
        plt.savefig(filename, format='pdf')
        plt.show()

def plot_mean_std(results, partitioning, bias_lower, bias_upper, rule_labels, marker_styles, output_dir, n_tosses):
    """
    Generate and save mean and standard deviation plots across thresholds.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, partition in enumerate(partitioning):
        current_partition_results = results[i]
        num_biases = partition
        bias_values = np.linspace(bias_lower, bias_upper, num_biases)

        title_str = f"Coin biases for H: {', '.join([f'{b:.2f}' for b in bias_values])}"
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(title_str, fontsize=16)

        for rule_idx, (rule_name, marker) in enumerate(zip(rule_labels, marker_styles)):
            all_rule_values = np.array([
                [bias_result[rule_idx] for bias_result in threshold_results]
                for threshold_results in current_partition_results
            ])
            mean_values = np.mean(all_rule_values, axis=0)
            std_values = np.std(all_rule_values, axis=0)

            ax.errorbar(bias_values, mean_values, yerr=std_values, label=rule_name,
                        capsize=5, marker=marker, linestyle='None', markersize=8)

        ax.set_xlabel('Bias')
        ax.set_ylabel('Values')
        ax.grid(True)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        filename = os.path.join(output_dir, f"n_tosses_{n_tosses}_partition_{partition}_mean_std_plot.pdf")
        plt.savefig(filename, format='pdf')
        plt.show()

def export_results_to_csv(results, partitioning, thresholds, bias_lower, bias_upper, rule_labels, output_dir, n_tosses):
    """
    Export simulation results to CSV files for each partition.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, partition in enumerate(partitioning):
        current_partition_results = results[i]
        num_biases = partition
        bias_values = np.linspace(bias_lower, bias_upper, num_biases)

        # Prepare the data for CSV
        csv_data = []
        for j, threshold_results in enumerate(current_partition_results):
            threshold = thresholds[j]
            for k, bias in enumerate(bias_values):
                row = [bias, threshold]
                row.extend([threshold_results[k][rule_idx] for rule_idx in range(len(rule_labels))])
                csv_data.append(row)

        # Define CSV filename
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = os.path.join(output_dir, f'n_tosses_{n_tosses}_partition_{partition}_results_{timestamp}.csv')

        # Write to CSV
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Bias', 'Threshold'] + rule_labels)
            writer.writerows(csv_data)

        print(f"Exported results for partition {partition} to {filename}.")
        