import pandas as pd

from fastprogress import progress_bar, master_bar

if __name__ == "__main__":
    mb = master_bar(["train", "test"])
    for phase in mb:
        df = pd.read_csv(f"{phase}.csv")
        dfs = []

        for filename in progress_bar(df.spectrum_filename, parent=mb):
            spectrum = pd.read_csv(
                f"spectrum/{filename}", sep="\t", header=None)
            spectrum.columns = ["wl", "intensity"]
            spectrum["spectrum_filename"] = filename
            dfs.append(spectrum)

        spectrums = pd.concat(dfs, axis=0).reset_index(drop=True)
        spectrums.to_csv(f"{phase}_spectrum.csv", index=False)
