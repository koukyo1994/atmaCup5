import pandas as pd

if __name__ == "__main__":
    for phase in ["train", "test"]:
        df = pd.read_csv(f"{phase}.csv")
        dfs = []

        for filename in df.spectrum_filename:
            spectrum = pd.read_csv(
                f"spectrum/{filename}", sep="\t", header=None)
            spectrum.columns = ["wl", "intensity"]
            spectrum["spectrum_filename"] = filename
            dfs.append(spectrum)

        spectrums = pd.concat(dfs, axis=0).reset_index(drop=True)
        spectrums.to_csv("train_spectrum.csv", index=False)
