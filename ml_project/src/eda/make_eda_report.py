import pandas as pd
from pandas_profiling import ProfileReport


RAW_DATA_PATH = './data/raw/heart_cleveland_upload.csv'
OUTPUT_PATH = './reports/EDA.html'


def make_eda_report(input_path=RAW_DATA_PATH, output_path=OUTPUT_PATH):
    raw_df = pd.read_csv(input_path)
    profile = ProfileReport(raw_df, title='EDA Report')
    profile.to_file(output_file=output_path)


if __name__ == "__main__":
    make_eda_report()
