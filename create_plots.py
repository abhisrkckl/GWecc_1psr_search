import sys
import json
import corner
import matplotlib.pyplot as plt

from analysis import read_ptmcmc_chain

def get_plt_param_name(param: str):
    param_map = {
        "red_noise_gamma" : "$\\gamma_{rn}$",
        "red_noise_log10_A" : "$\\log_{10} A_{rn}$",
        "gwecc_deltap": "$\\delta_p$",
        "gwecc_e0" : "$e_0$",
        "gwecc_eta": "$\\eta$",
        'gwecc_l0': "$l_0$",
        'gwecc_log10_A': "$\\log_{10} \\zeta_{0}$",
        'gwecc_log10_F': "$\\log_{10} f_{gw}$",
        'gwecc_log10_M': "$\\log_{10} M$",
        'gwecc_rho': "$\\rho$",
        'gwecc_sigma': "$\\sigma$"
    }
    for par, pltpar in param_map.items():
        if param.endswith(par):
            return pltpar
    
    raise ValueError(f"Parameter {param} not found.")

result_dir = sys.argv[1]
show_plots = len(sys.argv) >= 3 and sys.argv[2] == "show"

summary_file = f"{result_dir}/summary.json"
with open(summary_file, "rb") as sf:
    summary = json.load(sf)

if summary["sampler"] == "ptmcmc":
    chain = read_ptmcmc_chain(result_dir, summary["ptmcmc_burnin_fraction"])

param_names = list(summary["pta_param_summary"].keys())

corner.corner(chain, labels=param_names)
plt.savefig(f"{result_dir}/corner.pdf")
if show_plots:
    plt.show()

