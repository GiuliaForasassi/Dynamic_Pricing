import io

import matplotlib.pyplot as plt
import mlflow
import pandas as pd


def plot_and_log_figures(df: pd.DataFrame, log_to_mlflow: bool = True) -> None:
    # Save each Matplotlib graph in memory and upload it directly to MLflow as an image
    def log_fig(name_graph: str) -> None:
        if log_to_mlflow:
            mlflow.log_figure(plt.gcf(), f"figures/{name_graph}.png")
        plt.close()

    # ========== 1) Requests for each category ==========
    d = df[df["kind"] == "initial_requests"]
    if not d.empty:
        plt.figure()
        d.groupby("category", sort=False)["value"].sum().plot(kind="bar")
        plt.title("Requests for each category")
        plt.ylabel("Requests number")
        plt.tight_layout()
        log_fig("01_Initial_Requests")

    # ========== 2) Occupancy distribution ==========
    d = df[df["kind"] == "occupancy_dist_sample"]
    if not d.empty:
        plt.figure()
        plt.hist(d["value"].dropna(), bins=30)
        plt.title("Occupancy distribution")
        plt.xlabel("Occupancy (0-1)")
        plt.ylabel("Rate")
        plt.tight_layout()
        log_fig("02_occupancy_distribution")

    # ========== 3) Prices ==========

    # ========== 4) For each day: action curve ==========
    d = df[df["kind"] == "action_curve"].sort_values("date")
    if not d.empty:
        plt.figure()
        for cat, g in d.groupby("category", sort=False):
            plt.plot(g["date"], g["value"], label=str(cat))
        plt.title("Action curve)")
        plt.xlabel("Simulation day")
        plt.ylabel("Action")
        plt.legend()
        plt.tight_layout()
        log_fig("04_action_curve_network")

    # ========== 5) For each day: occupancy % ==========
    d = df[df["kind"] == "occupancy_pct"].sort_values("date")
    if not d.empty:
        plt.figure()
        for cat, g in d.groupby("category", sort=False):
            plt.plot(g["date"], g["value"], label=str(cat))
        plt.title("Occupancy percentage (days)")
        plt.xlabel("Simulation days")
        plt.ylabel("Occupancy (%)")
        plt.legend()
        plt.tight_layout()
        log_fig("05_occupancy_percentage_daily")

    # ========== 6) Advance.... ==========
    for k in (1, 15, 60):
        d = df[df["kind"] == f"advance_{k}"].sort_values("date")
        if not d.empty:
            plt.figure()
            for cat, g in d.groupby("category", sort=False):
                plt.plot(g["date"], g["value"], label=str(cat))
            plt.title(f"Advance {k} days")
            plt.xlabel("Simulation day")
            plt.ylabel("Price")
            plt.legend()
            plt.tight_layout()
            log_fig(f"06_advance_{k}_days")
