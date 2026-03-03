import numpy as np
import pandas as pd
import sys

def calc_quantiles (metrics, num):
    """
    calculate `num` quantiles for the given list
    """
    DEBUG = False

    bins = np.linspace(0, 1, num=num, endpoint=True)
    s = pd.Series(metrics)
    q = s.quantile(bins, interpolation="nearest")

    try:
        dig = np.digitize(metrics, q) - 1
    except ValueError as e:
        print("ValueError:", str(e), metrics, s, q, bins)
        sys.exit(-1)

    quantiles = []

    for idx, q_hi in q.items():
        quantiles.append(q_hi)

        if DEBUG:
            print(idx, q_hi)

    return quantiles


def calc_alpha_ptile (alpha_measures, show=True):
    """
    calculate the quantiles used to define a threshold alpha cutoff
    """
    quantiles = calc_quantiles(alpha_measures, num=10)
    num_quant = len(quantiles)

    if show:
        print("\tptile\talpha")

        for i in range(num_quant):
            percentile = i / float(num_quant)
            print("\t{:0.2f}\t{:0.4f}".format(percentile, quantiles[i]))

    return quantiles, num_quant


    