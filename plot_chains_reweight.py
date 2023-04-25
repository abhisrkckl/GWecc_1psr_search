import numpy as np
import matplotlib.pyplot as plt
import sys
import json
import corner

def get_param_name(pname: str):
    if "gwecc" in pname:
        known_names = ["log10_A", "sigma", "rho", "log10_F", "log10_M", "eta", "e0", "l0", "deltap"]
        latex_names = ["$\\log_{10} \\zeta_0$", "$\\sigma$", "$\\rho$", "$\\log_{10} f_{gw}$", 
                       "$\\log_{10} M$", "$\\eta$", "$e_0$", "$l_0$", "$\\Delta_p$"]
        for kn, ln in zip(known_names, latex_names):
            if pname.endswith(kn):
                return ln
    elif "red_noise" in pname:
        known_names = ["log10_A", "gamma"]
        latex_names = ["$\\log_{10} A_{RN}$", "$\\gamma_{RN}$"]
        for kn, ln in zip(known_names, latex_names):
            if pname.endswith(kn):
                return ln

def upper_limit_weight(value, pmin, pmax):
    mask = np.logical_and(value>=pmin, value<=pmax)
    uniform_prior = 1 / (pmax-pmin)
    linexp_prior = np.log(10) * 10**value / (10**pmax - 10**pmin)
    
    weight = mask * linexp_prior / uniform_prior
    weight /= sum(weight)

    return weight 

chaindir = sys.argv[1]

with open(f"{chaindir}/summary.json", "r") as sf:
    summary = json.load(sf)

param_names = list(summary["pta_param_summary"].keys())
param_names = list(map(get_param_name, param_names))

print(param_names)
 
chain = np.genfromtxt(f"{chaindir}/chain_1.txt")
burn = summary.get("burnin", 0)
burned_chain = chain[burn:, :-4]

ndim = len(param_names)
for idx, (ch, pn) in enumerate(zip(burned_chain.T, param_names)):
    plt.subplot(ndim, 1, idx+1)
    plt.plot(ch)
    plt.ylabel(pn)
plt.show()

if "burnin" in summary:
    filter_param_idxs = [idx for idx, pn in enumerate(param_names) if "RN" not in pn]
    filter_param_names = [pn for pn in param_names if "RN" not in pn]

    log10_A_idx = list(summary["pta_param_summary"].keys()).index("gwecc_log10_A")
    log10_A_samples = burned_chain[:,log10_A_idx]
    weights = upper_limit_weight(log10_A_samples, -11, -5)

    print(burned_chain[:,filter_param_idxs].shape, weights.shape)

    corner.corner(burned_chain[:,filter_param_idxs], weights=weights,
                  labels=filter_param_names, 
                  bins=16, max_n_ticks=3, smooth=0.5, label_kwargs={"fontsize": 12}, 
                  labelpad=0.4, color="blue")
    plt.show()