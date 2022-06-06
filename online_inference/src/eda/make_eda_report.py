import click
import pandas as pd
from pandas_profiling import ProfileReport

from src.configs import read_config


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def make_eda_report(config_path: str):
    configs = read_config(config_path)
    input_path = configs.input_data_path
    output_path = configs.output_model_path

    raw_df = pd.read_csv(input_path)
    profile = ProfileReport(raw_df, title='EDA Report')
    profile.to_file(output_file=output_path)


if __name__ == "__main__":
    make_eda_report()
