import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from matplotlib.container import BarContainer


def plot_and_log_figures(df: pd.DataFrame, log_to_mlflow: bool = True) -> None:
    # Save each Matplotlib graph in memory and upload it directly to MLflow as an image
    def log_fig(name_graph: str) -> None:
        if log_to_mlflow:
            mlflow.log_figure(plt.gcf(), f"figures/{name_graph}.png")
        plt.close()

    # ========== 1) Requests for each category ==========
    # Filter the original dataframe keeping only the rows about initial requests
    d = df[df["kind"] == "initial_requests"]
    if not d.empty:
        d = d.copy()
        d["category"] = d["category"].astype(int)
        s = d.groupby("category")["value"].sum().sort_index()
        ax = s.plot(kind="bar")
        ax.set(title="Requests for each category", xlabel="Category", ylabel="Number of requests")
        for c in ax.containers:  # type: ignore
            c: BarContainer
            ax.bar_label(c, fmt="%d", padding=2)
        plt.tight_layout()
        log_fig("1_Initial_Requests")

    # ========== 2A) Occupancy distribution ==========
    d = df[df["kind"] == "occupancy_dist"].copy()
    if not d.empty:
        d["value"] = pd.to_numeric(d["value"], errors="coerce")
        vals = d["value"]
        # DEBUG TO UNDERSTAND WHY THE OCCUPANCY IS STRANGE
        print(
            "N:",
            vals.size,
            " >0:",
            int((vals > 0).sum()),
            " min/max:",
            (vals.min() if vals.size else None, vals.max() if vals.size else None),
            " mean:",
            (vals.mean() if vals.size else None),
        )
        plt.figure()
        plt.hist(vals, bins=20, range=(0, 1), density=True)
        plt.title("Occupancy distribution")
        plt.xlabel("Occupancy")
        plt.ylabel("Density")
        plt.tight_layout()
        log_fig("2A_occupancy_distribution")

    # ========== 2B) Daily occupancy over time ==========
    d = df[df["kind"] == "occupancy_dist"].copy()
    if not d.empty:
        d["day"] = pd.to_numeric(d["day"], errors="coerce")
        d["value"] = pd.to_numeric(d["value"], errors="coerce")
        d = d.dropna(subset=["day", "value"])

        # Average across all categories for each day
        daily_mean = d.groupby("day", sort=True)["value"].mean()

        plt.figure()
        x = daily_mean.index.to_numpy(dtype=int)
        y = daily_mean.to_numpy(dtype=float)
        plt.plot(x, y, label="All categories (mean)")
        plt.title("Daily occupancy over time")
        plt.xlabel("Simulation day")
        plt.ylabel("Occupancy")
        plt.legend()
        plt.tight_layout()
        log_fig("2B_daily_occupancy_over_time")

        # One curve for each category
        pivot = (
            d.assign(category=d["category"].astype(str).str.strip())
            .pivot_table(index="day", columns="category", values="value", aggfunc="mean")
            .sort_index()
        )
        if pivot.shape[1] > 0:
            plt.figure()
            # Convert index to numpy array with explicit dtypes to avoid ExtensionArray issues
            x = pivot.index.to_numpy(dtype=int)
            for col in pivot.columns:
                # Convert column to plain numpy arrays with explicit dtypes to avoid ExtensionArray issues
                y = pivot[col].to_numpy(dtype=float)
                plt.plot(x, y, label=f"Cat {col}")
            plt.title("Daily occupancy per category")
            plt.xlabel("Simulation day")
            plt.ylabel("Occupancy")
            plt.legend()
            plt.tight_layout()
            log_fig("2C_daily_occupancy_per_category")

    # ========== 2D) For each day: occupancy % ==========
    d = df[df["kind"] == "occupancy_pct"].sort_values("day")
    if not d.empty:
        plt.figure()
        for cat, g in d.groupby("category", sort=False):
            plt.plot(g["day"], g["value"], label=str(cat))
        plt.title("Occupancy percentage (days)")
        plt.xlabel("Simulation days")
        plt.ylabel("Occupancy (%)")
        plt.legend()
        plt.tight_layout()
        log_fig("2D_occupancy_percentage_daily")

    # ========== 3) Acceptance ==========
    d_acc_all = df[
        (df["category"] == "all") & (df["kind"].isin(["accepted_count", "rejected_count"]))
    ].copy()
    if not d_acc_all.empty:
        accepted = float(d_acc_all.loc[d_acc_all["kind"] == "accepted_count", "value"].sum())
        rejected = float(d_acc_all.loc[d_acc_all["kind"] == "rejected_count", "value"].sum())
        total = accepted + rejected

        if total > 0:
            plt.figure()
            plt.pie(
                [accepted, rejected],
                labels=["Accepted", "Rejected"],
                autopct=lambda p: f"{p:.1f}%\n({round(p * total / 100.0)})",
                startangle=90,
            )
            plt.title("Requests acceptance")
            plt.tight_layout()
            log_fig("3_acceptance_counts")

    # ========== 4) For each day: action curve ==========
    d = df[df["kind"] == "action_curve"].sort_values("day")
    if not d.empty:
        plt.figure()
        for cat, g in d.groupby("category", sort=False):
            plt.plot(g["day"], g["value"], label=str(cat))
        plt.title("Action curve)")
        plt.xlabel("Simulation day")
        plt.ylabel("Action")
        plt.legend()
        plt.tight_layout()
        log_fig("4_action_curve_network")

    """
    # ========== 6) Advance.... ==========
    for k in (1, 15, 60):
        d = df[df["kind"] == f"advance_{k}"].sort_values("day")
        if not d.empty:
            plt.figure()
            for cat, g in d.groupby("category", sort=False):
                plt.plot(g["day"], g["value"], label=str(cat))
            plt.title(f"Advance {k} days")
            plt.xlabel("Simulation day")
            plt.ylabel("Price")
            plt.legend()
            plt.tight_layout()
            log_fig(f"5_advance_{k}_days")
"""
