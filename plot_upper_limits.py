import matplotlib.pyplot as plt
import numpy as np
import sys
import json

def upper_limit_weight(value, pmin, pmax):
    mask = np.logical_and(value>=pmin, value<=pmax)
    uniform_prior = 1 / (pmax-pmin)
    linexp_prior = np.log(10) * 10**value / (10**pmax - 10**pmin)
    
    weight = mask * linexp_prior / uniform_prior
    weight /= sum(weight)

    return weight 

def plot_upper_limit(
    param_names,
    chain,
    xparam_name,
    xparam_label,
    xparam_lims,
    xparam_bins=16,
    quantile=0.95,
):
    ampl_idx = param_names.index("gwecc_log10_A")
    freq_idx = param_names.index(xparam_name)

    chain_ampl = chain[:, ampl_idx]
    chain_freq = chain[:, freq_idx]

    log10_F_min, log10_F_max = xparam_lims
    log10_F_lins = np.linspace(log10_F_min, log10_F_max, xparam_bins + 1)
    log10_F_mins = log10_F_lins[:-1]
    log10_F_maxs = log10_F_lins[1:]
    log10_F_mids = (log10_F_mins + log10_F_maxs) / 2

    A_quant_Fbin = []
    for log10_fmin, log10_fmax in zip(log10_F_mins, log10_F_maxs):
        freq_mask = np.logical_and(chain_freq >= log10_fmin, chain_freq < log10_fmax)
        log10_A_Fbin_data = chain_ampl[freq_mask]
        A_quant = np.quantile(10**log10_A_Fbin_data, quantile)
        A_quant_Fbin.append(A_quant)

    A_quant = np.quantile(10**chain_ampl, quantile)

    plt.plot(log10_F_mids, A_quant_Fbin)
    plt.axhline(A_quant, color="grey")
    plt.yscale("log")
    plt.xlabel(xparam_label)
    plt.ylabel(f"{int(100*quantile)}%" + " upper bound on $\\zeta_0$ (s)")

if __name__ == "__main__":
    chain_dir = sys.argv[1]

    chain = np.genfromtxt(f"{chain_dir}/chain_1.txt")
    with open(f"{chain_dir}/summary.json", "r") as sf:
        summary = json.load(sf)

    param_names = list(summary["pta_param_summary"].keys())
    burn = summary["burnin"]

    burned_chain = chain[burn:, :-4]

    plt.subplot(121)
    plot_upper_limit(
        param_names,
        burned_chain,
        "gwecc_log10_F",
        "$\\log_{10} f_{gw}$ (Hz)",
        (-9,-7),
        xparam_bins=16,
        quantile=0.95,
    )

    # year = 3600*24*365.25
    # plt.axvline(np.log10(1/year))
    # plt.axvline(np.log10(2/year))
    # plt.axvline(np.log10(0.5/year))
    # plt.axvline(np.log10(0.25/year))

    plt.subplot(122)
    plot_upper_limit(
        param_names,
        burned_chain,
        "gwecc_e0",
        "$e_0$",
        (0.01, 0.8),
        xparam_bins=8,
        quantile=0.95,
    )

    plt.show()