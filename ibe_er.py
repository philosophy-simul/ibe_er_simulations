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
from typing import List, Dict, Tuple
import os
from matplotlib import pyplot as plt
import csv
import datetime
from itertools import combinations

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

# This gives factual evidence.

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
    observed_evidence = []
    for toss,unc in zip(sequence,uncertainties):
        if np.random.uniform(0,1)<=unc:
            observed_evidence.append(toss)
        else:
            observed_evidence.append(1-toss)
    return [{"toss": toss, "evidential certainty": certainty} for toss, certainty in zip(observed_evidence, uncertainties)]

# This gives associated uncertainty levels.

        
def certain_uncertain_evidence(uncertain_seq, threshold,observations):
    certain_evidence = []
    uncertain_evidence = []
    # Loop through each entry in the list and apply the flipping condition
    for entry,toss in zip(uncertain_seq,observations):
        evidCertainty = entry['evidential certainty']
        
        if evidCertainty >= threshold:
            certain_evidence.append(toss)

        else:
            uncertain_evidence.append(toss)

    return [certain_evidence,uncertain_evidence]



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

def likelihood_calibration(likelihoods_heads, certainty_heads):
    first_pass = [l*certainty_heads + (1-l)*(1-certainty_heads) for l in likelihoods_heads]
    second_pass = [l*certainty_heads + (1-l)*(1-certainty_heads) for l in first_pass]

    return second_pass
    
def best_expl_uncert(uncertain_seq, hyp_space) -> np.ndarray:
    observed_heads = [i["evidential certainty"] if i["toss"]==1 else 1-i["evidential certainty"] for i in uncertain_seq]
    total_observed_heads = sum(observed_heads)
    expected_observations = np.zeros(len(hyp_space))
    for outcome in uncertain_seq:
        expected_observations+= np.array(likelihood_calibration(hyp_space,outcome["evidential certainty"]))
        # Compute absolute differences
    differences = np.abs(expected_observations - total_observed_heads)
    
    # Find the minimum difference
    min_diff = np.min(differences)
    
    # Get all indices where the difference equals the minimum
    closest_indices = np.where(np.isclose(differences, min_diff))[0]
    
    return closest_indices




def jc_ibe_star(uncertain_seq, priorJC=None, hyp_space=None, bonus=0.1):
    if hyp_space is None:
        hyp_space = [1/6, 2/6, 3/6, 4/6, 5/6]
    if priorJC is None:
        priorJC = [1 / len(hyp_space)] * len(hyp_space)  # Uniform prior by default
    priorIBE = priorJC

    posteriorJC = []
    posteriorIBE = []

    distrsJC = [priorJC[:]]
    distrsIBE = [priorJC[:]]

    observed_tosses = []
    uncertain_seq2=[]
    for outcome in uncertain_seq:
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
        uncertain_seq2.append(outcome)
        likelihoods = likelihood_calibration(hyp_space,p_e_post)
        if toss!=1:
            likelihoods=[1-i for i in likelihoods]
        likelihoods=np.array(likelihoods)
        # JC Update
        p_h_and_e = np.array([p_h_prior * p_e_given_h for p_h_prior, p_e_given_h in zip(priorJC, likelihoods)])
        p_h_and_neg_e = np.array([p_h_prior * (1 - p_e_given_h) for p_h_prior, p_e_given_h in zip(priorJC, likelihoods)])
        p_e_prior = np.sum(p_h_and_e)
        p_h_given_e = p_h_and_e / p_e_prior
        p_h_given_neg_e = p_h_and_neg_e / (1 - p_e_prior)
        posteriorJC = p_e_post * p_h_given_e + (1 - p_e_post) * p_h_given_neg_e

        # IBE Update
        IBEindex = best_expl_uncert(uncertain_seq2, hyp_space)
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

def expected_score_prob(distr,bonus,penalty):
    options=range(len(distr))
    # Generate all non-empty subsets
    all_subsets = []
    
    for r in range(1, len(options) + 1):
        for subset in combinations(options, r):
            all_subsets.append(list(subset))
    
    subset_sums = [sum(distr[i] for i in subset) for subset in all_subsets]
    
    eus = []
    for option,pr in zip(all_subsets,subset_sums):
        eus.append(pr*((len(distr) - len(option)) / (len(distr) - 1)) * bonus - (1-pr)*penalty)
    return all_subsets[eus.index(max(eus))]



def single_experiment(
    true_bias: float,
    nr_tosses: int,
    thresholds: List[float],
    hyp_space: List[float],
    lower_bound: float = 0.5,
    upper_bound: float = 1.0,
    bonus: float = 0.1,
    weight_positive=1,
    weight_negative=1
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
    threshold_results=[]
    observations = [uncertain_evidence[i]["toss"] for i in range(len(uncertain_evidence))]
    for threshold in thresholds:
        certain_uncertain_ev=certain_uncertain_evidence(uncertain_evidence,threshold,observations)
        # ibe ER
        certain_partition=certain_uncertain_ev[0]
        uncertain_partition = certain_uncertain_ev[1]
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
        
        # ibe Standard: gets the observed evidence, explains it
        ibe_standard = best_expl(certain_partition+uncertain_partition,hyp_space)
        # ibe Standard-filtered: only explains the certain part, disregards the uncertain evidence
        if certain_partition!=[]:
            ibe_standardFiltered = best_expl(certain_partition,hyp_space)
        else:
            ibe_standardFiltered=np.array(hyp_space)
            # if there is no certain evidence, just make no inference, i.e. infer a disjunction of every option.
            
        # JC-IBE-Star inference
        prob=jc_ibe_star(uncertain_evidence,None,hyp_space,bonus)
        jc_result=np.array([hyp_space[i] for i in expected_score_prob(prob[0][-1], weight_positive, weight_negative)])
        ibe_star_result=np.array([hyp_space[i] for i in expected_score_prob(prob[1][-1], weight_positive, weight_negative)])
        threshold_results.append([ibe_er_result, ibe_standard,ibe_standardFiltered,jc_result,ibe_star_result])
    return threshold_results

    
def simulation(
    nr_repetitions, true_bias, nr_tosses, thresholds, 
    weight_positive=1, weight_negative=1, hyp_space=[1/6, 2/6, 3/6, 4/6, 5/6], 
    lower_bound=0.5, upper_bound=1,bonus=0.1
):

    thresh_scores=[]
    for run in range(nr_repetitions):
        i=0
        res = single_experiment(true_bias, nr_tosses, thresholds, hyp_space, lower_bound, upper_bound,bonus,weight_positive,weight_negative)
        thresh_scores_temp=[]
        for thresh in res:
            ibeer=0
            ibestand=0
            ibestandFilt=0
            jc=0
            ibest=0
            if true_bias in thresh[0]:
                ibeer+=((len(hyp_space) - len(thresh[0])) / (len(hyp_space) - 1)) * weight_positive
            else:
                ibeer-=weight_negative
    
            if true_bias in thresh[1]:
                ibestand+=((len(hyp_space) - len(thresh[1])) / (len(hyp_space) - 1)) * weight_positive
            else:
                ibestand-=weight_negative
    
            if true_bias in thresh[2]:
                ibestandFilt+=((len(hyp_space) - len(thresh[2])) / (len(hyp_space) - 1)) * weight_positive
            else:
                ibestandFilt-=weight_negative
            if true_bias in thresh[3]:
                jc+=((len(hyp_space) - len(thresh[3])) / (len(hyp_space) - 1)) * weight_positive
            else:
                jc-=weight_negative
            if true_bias in thresh[4]:
                ibest+=((len(hyp_space) - len(thresh[4])) / (len(hyp_space) - 1)) * weight_positive
            else:
                ibest-=weight_negative
            thresh_scores_temp.append([ibeer,ibestand,ibestandFilt,jc,ibest])

            i+=1
        thresh_scores.append(thresh_scores_temp)
        
# Initialize result containers
    avg_thresh = []
    std_thresh = []
    
    for run_idx in range(len(thresh_scores[0])):  # for each parameter
        param_avg = []
        param_std = []
        for metric_idx in range(len(thresh_scores[0][run_idx])):  # for each metric
            # Get all values for this parameter-metric pair
            values = [run[run_idx][metric_idx] for run in thresh_scores]
            mean = sum(values) / nr_repetitions
            variance = sum((x - mean) ** 2 for x in values) / (nr_repetitions - 1) if nr_repetitions > 1 else 0
            stddev = np.sqrt(variance)
            param_avg.append(mean)
            param_std.append(stddev)
        avg_thresh.append(param_avg)
        std_thresh.append(param_std)
    
    # Determine the number of metric positions (2 in this case)
    num_metrics = len(thresh_scores[0][0])
    
    all_values = [[] for _ in range(num_metrics)]  # Create a list for each metric position
    
    # Collect all values by metric index
    for run in thresh_scores:
        for param in run:
            for i, metric in enumerate(param):
                all_values[i].append(metric)
    
    # Compute mean and std for each metric
    avg_overall = [np.mean(values) for values in all_values]
    std_overall = [np.std(values, ddof=1) for values in all_values]


    return avg_thresh,std_thresh,avg_overall,std_overall,thresh_scores,all_values


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




def run_simulations(n_tosses,n_repetitions=1000, bias_lower=.1, bias_upper=.9, partitioning=[3,5,7,9,11], thresholds=[0.5,0.6,.7,.8,.9,1],weight_positive=1,weight_negative=1,uncert_lower=0.5,uncert_upper=1.0,bonus=0.1):
    """
    Run simulations for different partitions, thresholds, and biases.
    """
    results = []
    for partition in partitioning:
        print(f"Partition: {partition}")
        partition_results = []

        for bias in np.linspace(bias_lower,bias_upper,partition)[int(len(np.linspace(bias_lower,bias_upper,partition))/2):]:
            print(f"Bias: {bias}")
            partition_results.append(
                simulation(n_repetitions, bias, n_tosses, thresholds, weight_positive, weight_negative, 
                           np.linspace(bias_lower, bias_upper, partition), uncert_lower, uncert_upper,bonus)
            )
        results.append(partition_results)
    return results

def plot_avg_thresh_std(results, partitioning, thresholds, bias_lower, bias_upper, rule_labels, marker_styles, output_dir, n_tosses):
    """
    Plot avg_thresh and std_thresh results for each threshold, across biases and rules.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, partition in enumerate(partitioning):
        current_partition_results = results[i]
        num_biases = partition
        bias_values = np.linspace(bias_lower, bias_upper, num_biases)[int(len(np.linspace(bias_lower, bias_upper, num_biases)) / 2):]
        jitter = ((max(bias_values) - min(bias_values)) / (len(bias_values) - 1)) * 0.25

        for threshold_idx, threshold in enumerate(thresholds):
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.suptitle(f'Partition: {partition}, Threshold: {threshold:.1f}', fontsize=16)
            position = -1

            for rule_idx, (rule_name, marker) in enumerate(zip(rule_labels, marker_styles)):
                means = [bias_result[0][threshold_idx][rule_idx] for bias_result in current_partition_results]
                stds = [bias_result[1][threshold_idx][rule_idx] for bias_result in current_partition_results]

                bias_values_jittered = [val + jitter * position for val in bias_values]
                ax.errorbar(bias_values_jittered, means, yerr=stds, label=rule_name, marker=marker,
                            linestyle='None', markersize=8, capsize=5)
                position += 0.5

            ax.set_xlabel('Bias')
            ax.set_ylabel('Avg Value ± Std')
            ax.set_xticks(bias_values)
            ax.set_xticklabels([f"{bias:.2f}" for bias in bias_values])
            bias_val_dif = bias_values[1] - bias_values[0]
            grid_values = [val + bias_val_dif / 2 for val in bias_values[:-1]]
            for gv in grid_values:
                ax.axvline(gv, color='black', linestyle='--', linewidth=0.7)
            ax.axhline(0, color='lightgray', linestyle='--', linewidth=0.7)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            filename = os.path.join(output_dir, f"n_tosses_{n_tosses}_partition_{partition}_threshold_{threshold:.1f}_avg_thresh_std.pdf")
            plt.savefig(filename, format='pdf')
            plt.show()

def plot_avg_overall_std(results, partitioning, bias_lower, bias_upper, rule_labels, marker_styles, output_dir, n_tosses):
    """
    Plot avg_overall and std_overall results across biases and rules (ignoring thresholds).
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, partition in enumerate(partitioning):
        current_partition_results = results[i]
        num_biases = partition
        bias_values = np.linspace(bias_lower, bias_upper, num_biases)[int(len(np.linspace(bias_lower, bias_upper, num_biases)) / 2):]
        jitter = ((max(bias_values) - min(bias_values)) / (len(bias_values) - 1)) * 0.25

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(f'Partition: {partition} - Overall Averages', fontsize=16)
        position = -1

        for rule_idx, (rule_name, marker) in enumerate(zip(rule_labels, marker_styles)):
            means = [bias_result[2][rule_idx] for bias_result in current_partition_results]
            stds = [bias_result[3][rule_idx] for bias_result in current_partition_results]

            bias_values_jittered = [val + jitter * position for val in bias_values]
            ax.errorbar(bias_values_jittered, means, yerr=stds, label=rule_name, marker=marker,
                        linestyle='None', markersize=8, capsize=5)
            position += 0.5

        ax.set_xlabel('Bias')
        ax.set_ylabel('Avg Overall Value ± Std')
        ax.set_xticks(bias_values)
        ax.set_xticklabels([f"{bias:.2f}" for bias in bias_values])
        bias_val_dif = bias_values[1] - bias_values[0]
        grid_values = [val + bias_val_dif / 2 for val in bias_values[:-1]]
        for gv in grid_values:
            ax.axvline(gv, color='black', linestyle='--', linewidth=0.7)
        ax.axhline(0, color='lightgray', linestyle='--', linewidth=0.7)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        filename = os.path.join(output_dir, f"n_tosses_{n_tosses}_partition_{partition}_avg_overall_std.pdf")
        plt.savefig(filename, format='pdf')
        plt.show()

def plot_mean_std_by_threshold_over_bias(results, partitioning, thresholds, bias_lower, bias_upper, rule_labels, marker_styles, output_dir, n_tosses):
    """
    Plot mean/std results per threshold (x-axis), averaged over biases for each rule.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, partition in enumerate(partitioning):
        current_partition_results = results[i]
        num_biases = partition

        # Collect mean/std across biases for each threshold and rule
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(f'Partition: {partition} - Mean over Biases by Threshold', fontsize=16)
        position = -1

        for rule_idx, (rule_name, marker) in enumerate(zip(rule_labels, marker_styles)):
            means = []
            stds = []

            for threshold_idx in range(len(thresholds)):
                # Get all values for this threshold across biases
                values = [bias_result[0][threshold_idx][rule_idx] for bias_result in current_partition_results]
                means.append(np.mean(values))
                stds.append(np.std(values, ddof=1))

            thresholds_jittered = [t + 0.01 * position for t in thresholds]
            ax.errorbar(thresholds_jittered, means, yerr=stds, label=rule_name, marker=marker,
                        linestyle='None', markersize=8, capsize=5)
            position += 0.5

        ax.set_xlabel('Threshold')
        ax.set_ylabel('Avg Value ± Std (over biases)')
        ax.set_xticks(thresholds)
        ax.axhline(0, color='lightgray', linestyle='--', linewidth=0.7)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        filename = os.path.join(output_dir, f"n_tosses_{n_tosses}_partition_{partition}_mean_by_threshold_over_bias.pdf")
        plt.savefig(filename, format='pdf')
        plt.show()

def plot_mean_std_overall(results, partitioning, rule_labels, marker_styles, output_dir, n_tosses):
    """
    Plot overall average (and std) across all thresholds and biases for each rule,
    using different markers and default colors per column.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, partition in enumerate(partitioning):
        current_partition_results = results[i]
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(f'Partition: {partition} - Overall Mean and Std (all biases and thresholds)', fontsize=16)

        means = np.mean([bias_result[2] for bias_result in current_partition_results], axis=0)
        stds = np.mean([bias_result[3] for bias_result in current_partition_results], axis=0)

        positions = range(len(rule_labels))
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']  # Use matplotlib default color cycle

        for idx, (pos, mean, std, label) in enumerate(zip(positions, means, stds, rule_labels)):
            ax.errorbar(
                pos, mean, yerr=std,
                fmt=marker_styles[idx % len(marker_styles)],
                color=color_cycle[idx % len(color_cycle)],
                linestyle='None', capsize=5, markersize=8, label=label
            )

        ax.set_xticks(list(positions))
        ax.set_xticklabels(rule_labels)
        ax.set_ylabel('Overall Avg Value ± Std')
        ax.axhline(0, color='lightgray', linestyle='--', linewidth=0.7)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        filename = os.path.join(output_dir, f"n_tosses_{n_tosses}_partition_{partition}_overall_mean_std.pdf")
        plt.savefig(filename, format='pdf')
        plt.show()    
        



def export_all_results_to_csv(results, partitioning, thresholds, bias_lower, bias_upper, rule_labels, n_tosses, n_repetitions, weight_positive, weight_negative, uncert_lower, uncert_upper):
    """
    Export all simulation results: per-threshold stats and overall stats.
    Each partition gets two CSVs: one for threshold-level and one for overall-level results.
    """
    output_dir = (
        f"csvs_t{n_tosses}_r{n_repetitions}_wp{weight_positive}_wn{weight_negative}_"
        f"ul{uncert_lower:.2f}_uu{uncert_upper:.2f}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    for i, partition in enumerate(partitioning):
        current_partition_results = results[i]
        num_biases = partition
        bias_values = np.linspace(bias_lower, bias_upper, num_biases)[int(len(np.linspace(bias_lower, bias_upper, num_biases)) / 2):]

        # === Threshold-level results ===
        thresh_csv_data = []
        for bias_idx, bias_result in enumerate(current_partition_results):
            avg_thresh, std_thresh, _, _, _, _ = bias_result

            for threshold_idx, threshold in enumerate(thresholds):
                row = [bias_values[bias_idx], threshold]
                for rule_idx in range(len(rule_labels)):
                    row.append(avg_thresh[threshold_idx][rule_idx])
                    row.append(std_thresh[threshold_idx][rule_idx])
                thresh_csv_data.append(row)

        thresh_header = ['Bias', 'Threshold']
        for rule in rule_labels:
            thresh_header += [f'{rule} Mean', f'{rule} Std']

        thresh_filename = os.path.join(
            output_dir, f'n_tosses_{n_tosses}_partition_{partition}_threshold_results_{timestamp}.csv'
        )
        with open(thresh_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(thresh_header)
            writer.writerows(thresh_csv_data)
        print(f"Exported threshold-level results for partition {partition} to {thresh_filename}.")

        # === Overall results ===
        overall_csv_data = []
        for bias_idx, bias_result in enumerate(current_partition_results):
            _, _, avg_overall, std_overall, _, _ = bias_result
            row = [bias_values[bias_idx]]
            for rule_idx in range(len(rule_labels)):
                row.append(avg_overall[rule_idx])
                row.append(std_overall[rule_idx])
            overall_csv_data.append(row)

        overall_header = ['Bias']
        for rule in rule_labels:
            overall_header += [f'{rule} Overall Mean', f'{rule} Overall Std']

        overall_filename = os.path.join(
            output_dir, f'n_tosses_{n_tosses}_partition_{partition}_overall_results_{timestamp}.csv'
        )
        with open(overall_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(overall_header)
            writer.writerows(overall_csv_data)
        print(f"Exported overall results for partition {partition} to {overall_filename}.")

def adjust_thresholds(thresholds, uncert_lower, uncert_upper):
    thresholds = sorted(thresholds)  # ensure thresholds are sorted
    
    below_or_equal = [t for t in thresholds if t <= uncert_lower]
    above_or_equal = [t for t in thresholds if t >= uncert_upper]
    between = [t for t in thresholds if uncert_lower < t < uncert_upper]

    result = []
    if below_or_equal:
        result.append(below_or_equal[-1])  # keep the highest ≤ uncert_lower
    result += between
    if above_or_equal:
        result.append(above_or_equal[0])  # keep the lowest ≥ uncert_upper

    return result


def run_config(n_tosses, n_repetitions, partitioning, thresholds, bias_lower, bias_upper,
               weight_positive, weight_negative, uncert_lower, uncert_upper,
               rule_labels, marker_styles):
    print(f"Started: t={n_tosses}, wp={weight_positive}, wn={weight_negative}, ul={uncert_lower}, uu={uncert_upper}", flush=True)
    # Adjust thresholds for bounds
    thresholds = adjust_thresholds(thresholds, uncert_lower, uncert_upper)

    output_dir = (
        f"plots_t{n_tosses}_r{n_repetitions}_wp{weight_positive}_wn{weight_negative}_"
        f"ul{uncert_lower:.2f}_uu{uncert_upper:.2f}"
    )

    results = run_simulations(
        n_tosses=n_tosses,
        n_repetitions=n_repetitions,
        bias_lower=bias_lower,
        bias_upper=bias_upper,
        partitioning=partitioning,
        thresholds=thresholds,
        weight_positive=weight_positive,
        weight_negative=weight_negative,
        uncert_lower=uncert_lower,
        uncert_upper=uncert_upper
    )

    plot_avg_thresh_std(results, partitioning, thresholds, bias_lower, bias_upper,
                        rule_labels, marker_styles, output_dir, n_tosses)

    plot_avg_overall_std(results, partitioning, bias_lower, bias_upper,
                         rule_labels, marker_styles, output_dir, n_tosses)

    plot_mean_std_by_threshold_over_bias(results, partitioning, thresholds, bias_lower, bias_upper,
                                         rule_labels, marker_styles, output_dir, n_tosses)

    plot_mean_std_overall(results, partitioning, rule_labels, marker_styles, output_dir, n_tosses)

    export_all_results_to_csv(results, partitioning, thresholds, bias_lower, bias_upper,
                              rule_labels, n_tosses, n_repetitions,
                              weight_positive, weight_negative,
                              uncert_lower, uncert_upper)

    return f"Finished: t={n_tosses}, wp={weight_positive}, wn={weight_negative}, ul={uncert_lower}, uu={uncert_upper}"
