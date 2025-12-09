import numpy as np
import matplotlib.pyplot as plt

from ff_plot import run_ff_sweep

qs = [0.1, 0.5, 1.0]
NR_list = np.logspace(2, 3, num=6)
NR_list = np.unique(NR_list.astype(int))
seeds = [11, 23, 42, 84]

print("Running simulations...")
sim_runs = [
    run_ff_sweep(
        N_s=102,
        NR_list=NR_list,
        qs=qs,
        f_target=0.2,
        P=10_000,
        V=5,
        deltaS=0.2,
        seed=seed,
    )
    for seed in seeds
]

print("Creating plot...")
fig, axs = plt.subplots(3, len(qs), figsize=(14, 9), sharex=True)
ylabels = (r"$f^R$", r"$a^R$", "Dimension")
NR_plot = NR_list.astype(float)
xticks = [100, 158, 251, 398, 630, 1000]
xtick_labels = [r"$10^2$", "", "", "", "", r"$10^3$"]

for col, q in enumerate(qs):
    per_seed_f = []
    per_seed_a = []
    per_seed_d = []
    for run in sim_runs:
        per_q = run[float(q)]
        per_seed_f.append([per_q[int(n)].fR for n in NR_list])
        per_seed_a.append([per_q[int(n)].aR for n in NR_list])
        per_seed_d.append([per_q[int(n)].dim for n in NR_list])
    per_seed_f = np.array(per_seed_f)
    per_seed_a = np.array(per_seed_a)
    per_seed_d = np.array(per_seed_d)

    for row, (values, ylabel) in enumerate(zip((per_seed_f, per_seed_a, per_seed_d), ylabels)):
        ax = axs[row, col]
        mean_vals = values.mean(axis=0)
        std_vals = values.std(axis=0)
        ax.plot(
            NR_plot,
            mean_vals,
            color="tab:blue",
            linewidth=2.0,
            label="FF (mean)" if col == 0 and row == 0 else None,
        )
        ax.fill_between(
            NR_plot,
            mean_vals - std_vals,
            mean_vals + std_vals,
            color="tab:blue",
            alpha=0.2,
        )
        for seed_idx, seed_vals in enumerate(values):
            ax.scatter(
                NR_plot,
                seed_vals,
                color="tab:blue",
                edgecolor="white",
                linewidth=0.5,
                s=35,
                alpha=0.8,
                label="FF (samples)" if seed_idx == 0 and col == 0 and row == 0 else None,
            )
        ax.set_xscale("log")
        if row == 2:
            ax.set_yscale("log")
        if col == 0:
            ax.set_ylabel(ylabel)
        if row == 2:
            ax.set_xlabel(r"$N^R$")
        if row == 0:
            ax.set_title(f"q = {q}", fontsize=12, pad=10)
        if row == 0:
            ax.set_ylim(0, 0.3)
        if row == 1:
            ax.set_ylim(0, 0.15)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)

for ax in axs[-1]:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)

handles, labels = axs[0, 0].get_legend_handles_labels()
if handles:
    axs[0, 0].legend(loc="lower left", frameon=True)

fig.suptitle("Feedforward response metrics vs connectivity", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig("fig_page17_ff_reproduction.png", dpi=300)
plt.show()
print("Plot saved as fig_page17_ff_reproduction.png")

