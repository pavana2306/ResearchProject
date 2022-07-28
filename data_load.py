import pandas as pd

class DataFrame_Loader:
    def __init__(self):
        print("Loading DataFrame")

    def load_data_files(self, filepath):
        dftrain = pd.read_csv(filepath, error_bad_lines=True, sep='\t')
        return dftrain


load = DataFrame_Loader()
df = load.load_data_files("dataset/drugsComTest_raw.tsv")
print(df.head())

# We apply code preprocessing here.