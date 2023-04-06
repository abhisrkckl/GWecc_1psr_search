import matplotlib.pyplot as plt

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

    log10_A_quant_Fbin = []
    for log10_fmin, log10_fmax in zip(log10_F_mins, log10_F_maxs):
        freq_mask = np.logical_and(chain_freq >= log10_fmin, chain_freq < log10_fmax)
        log10_A_Fbin_data = chain_ampl[freq_mask]
        log10_A_quant = np.quantile(log10_A_Fbin_data, quantile)
        log10_A_quant_Fbin.append(log10_A_quant)

    log10_A_quant = np.quantile(chain_ampl, quantile)

    plt.plot(log10_F_mids, log10_A_quant_Fbin)
    plt.axhline(log10_A_quant, color="grey")
    plt.xlabel(xparam_label)
    plt.ylabel(f"{int(100*quantile)}%" + " upper bound on $\\log_{10} \\zeta_0$ (s)")