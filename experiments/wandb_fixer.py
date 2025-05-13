import wandb
from wandb.apis.public import Run
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def wandb_histogram_to_plt(run: Run):
    """
    Convert a wandb histogram to a matplotlib histogram.
    """

    # Get the histogram data from wandb

    hist = run.history()["image_histogram"].dropna().iloc[0]

    print("histogram:", hist)

    bin_meta = hist["packedBins"]
    bins = np.arange(
        bin_meta["min"], bin_meta["size"] * (bin_meta["count"] + 1), bin_meta["size"]
    )
    counts = hist["values"]

    plt.stairs(counts, bins, fill=True, color="blue", alpha=0.5)


if __name__ == "__main__":
    from pprint import pprint

    RUN_ID = "jzv14zzz"
    NAMESPACE = "alamst-kth-royal-institute-of-technology/headless-runs"

    wandb.login()

    run = wandb.Api().run(f"{NAMESPACE}/{RUN_ID}")

    history = run.history()

    print("run keys:")
    pprint(list(history.columns))

    histogram_keys = [col for col in history.columns if "histogram" in col]
    histogram_entries: pd.DataFrame = history[histogram_keys]

    print("entries:", histogram_entries)
    for _, entry in histogram_entries.iterrows():
        for col in histogram_keys:
            if entry[col] is not None:
                print(f"col: {col}")
                pprint(entry[col])

    wandb_histogram_to_plt(run)

    plt.show()

    #
    # hist = run.history()
    #
    # histogram_runs = hist["image_histogram"]
    #
    # print(hist)
