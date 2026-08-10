from pathlib import Path

import numpy as np
import pandas as pd


def convert_digitized_data(input_file="./data_files/tau_profiles/wpd_datasets.csv", output_directory="./data_files/tau_profiles/"):
    """
    Convert the WebPlotDigitizer CSV into separate Blue and Green text files.

    For each sample:
      - Theta is the mean of the median, upper-error and lower-error X values.
      - The three Y values are retained unchanged.
      - Output columns are:
            Theta (arcmin)
            tau (median)
            tau (upper error)
            tau (lower error)
    """

    input_file = Path(input_file)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # The first two rows contain dataset names and X/Y labels.
    # The numerical data begin on the third row.
    data = pd.read_csv(input_file, skiprows=2, header=None)

    if data.shape[1] < 12:
        raise ValueError(f"Expected at least 12 numerical columns, but found {data.shape[1]}.")

    # Column ordering in the input CSV:
    #
    #  0, 1 : Blue median X, Y
    #  2, 3 : Blue upper-error X, Y
    #  4, 5 : Blue lower-error X, Y
    #  6, 7 : Green median X, Y
    #  8, 9 : Green upper-error X, Y
    # 10,11 : Green lower-error X, Y

    blue = make_output_table(
        median_x=data.iloc[:, 0],
        median_y=data.iloc[:, 1],
        upper_x=data.iloc[:, 2],
        upper_y=data.iloc[:, 3],
        lower_x=data.iloc[:, 4],
        lower_y=data.iloc[:, 5],
    )

    green = make_output_table(
        median_x=data.iloc[:, 6],
        median_y=data.iloc[:, 7],
        upper_x=data.iloc[:, 8],
        upper_y=data.iloc[:, 9],
        lower_x=data.iloc[:, 10],
        lower_y=data.iloc[:, 11],
    )

    blue_file = output_directory / "digitized_obs_data_blue.txt"
    green_file = output_directory / "digitized_obs_data_green.txt"

    write_output_file(blue, blue_file)
    write_output_file(green, green_file)

    print(f"Saved {blue_file}")
    print(f"Saved {green_file}")


def make_output_table(
    median_x,
    median_y,
    upper_x,
    upper_y,
    lower_x,
    lower_y,
):
    """
    Construct one sample table.

    Rows containing incomplete values in any of the six required input
    columns are removed.
    """

    values = pd.DataFrame(
        {
            "median_x": pd.to_numeric(median_x, errors="coerce"),
            "median_y": pd.to_numeric(median_y, errors="coerce"),
            "upper_x": pd.to_numeric(upper_x, errors="coerce"),
            "upper_y": pd.to_numeric(upper_y, errors="coerce"),
            "lower_x": pd.to_numeric(lower_x, errors="coerce"),
            "lower_y": pd.to_numeric(lower_y, errors="coerce"),
        }
    ).dropna()

    theta = values[["median_x", "upper_x", "lower_x"]].mean(axis=1)

    output = pd.DataFrame(
        {
            "Theta (arcmin)": theta,
            "tau (median)": values["median_y"],
            "tau (upper error)": values["upper_y"],
            "tau (lower error)": values["lower_y"],
        }
    )

    return output.reset_index(drop=True)


def write_output_file(data, output_file):
    """
    Write a whitespace-separated text file with a commented header.
    """

    np.savetxt(output_file, data.to_numpy(), header="Theta (arcmin) tau (median) tau (upper error) tau (lower error)", comments="# ", fmt="%.10e")


if __name__ == "__main__":
    convert_digitized_data(input_file="./data_files/tau_profiles/digitized_observed_data.csv", output_directory="./data_files/tau_profiles/")