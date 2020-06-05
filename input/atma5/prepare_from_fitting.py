import numpy as np
import pandas as pd

from pathlib import Path

from fastprogress import progress_bar, master_bar
from scipy.stats import cauchy

if __name__ == "__main__":
    mb = master_bar(["train", "test"])
    output_dir = Path("spectrum_from_fitting")
    output_dir.mkdir(exist_ok=True, parents=True)

    fitting = pd.read_csv("fitting.csv")
    for phase in mb:
        df = pd.read_csv(f"{phase}.csv")

        df = df.merge(fitting, how="left", on="spectrum_id")

        for _, row in progress_bar(df.iterrows(), total=len(df), parent=mb):
            filename = row["spectrum_filename"]

            A0 = row["params1"]
            x0 = row["params2"]
            w0 = row["params3"]

            A1 = row["params4"]
            x1 = row["params5"]
            w1 = row["params6"]

            spec = pd.read_csv(f"spectrum/{filename}", sep="\t", header=None)
            intensity = spec[1].values
            wavelength = spec[0].values
            max_wv = wavelength.max()
            min_wv = wavelength.min()

            y0 = A0 * cauchy.pdf(wavelength, x0, w0)
            y1 = A1 * cauchy.pdf(wavelength, x1, w1)
            y = y0 + y1

            ratio = max(y) / intensity.max()
            y = y / ratio

            spec[1] = y
            spec.to_csv(
                output_dir / filename, sep="\t", header=None, index=False)
