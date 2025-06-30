#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation of biased coin tosses and inference using various evidence-based methods

Started on Tue Nov 12 17:06:46 2024
"""
import multiprocessing
from functions import run_config


# Main Code
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_repetitions = 1000
    partitioning = [3, 7, 11]
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    bias_lower = 0
    bias_upper = 1.0
    rule_labels = ['IBE-ER', 'IBE-Standard', 'IBE-StandardFiltered', 'JC', 'IBE-Star']
    marker_styles = ['o', 's', 'D', '^', '*']

    tasks = []
    for n_tosses in [10, 25, 50, 100]:
        for uncert_lower in [0.5]: # Here, one could add other (even multiple) values to change the span of uncertainty associated with uncertain evidence by setting the lower evidential confidence thresholds.
            for uncert_upper in [1.0]: # Here, one would set the upper thresholds.
                if uncert_lower >= uncert_upper:
                    continue
                for weight_positive in [0, 1, 2, 10]:
                    for weight_negative in [0, 1, 2, 10]:
                        if weight_positive == weight_negative and weight_positive != 1:
                            continue
                        tasks.append((
                            n_tosses, n_repetitions, partitioning, thresholds, bias_lower, bias_upper,
                            weight_positive, weight_negative, uncert_lower, uncert_upper,
                            rule_labels, marker_styles
                        ))

    print(f"Total runs: {len(tasks)}")

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_config, *args) for args in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                print(f"{i}/{len(tasks)} - {result}")
            except Exception as e:
                print(f"Error in task {i}: {e}")


# started running: 16:55