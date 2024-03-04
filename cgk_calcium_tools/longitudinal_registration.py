import json
import os
import pandas as pd
import isx
import unittest


def longitudinal_registration(
    input_cell_set_files: list,
    output_cell_set_files: list,
    json_file: str,
    overwrite: bool = True,
    verbose: bool = False,
    **kws,
) -> bool:
    """
    Utilize the isx.longitudinal_registration function to generate a JSON file
    with the obtained information for further processing

    Parameters
    ----------
    input_cell_set_files : list
        cell set files
    output_cell_set_files : list
        output cell set files. Has to hace the same side of input_cell_set_files
    input_movie_files : list, optional
        movie files
    json_file : str
        json file path
    overwrite : bool, optional
            Remove results and recompute them, by default False
    verbose : bool, optional
            Show additional messages, by default False

    Returns
    -------
    Bool
        Returns False if the JSON file was recomputed, possibly due to
        another file with the same name that could not be overwritten
    """
    if os.path.exists(json_file):
        if not overwrite:
            return False

    csv_file = os.path.splitext(json_file)[0] + ".csv"
    isx.longitudinal_registration(
        input_cell_set_files, output_cell_set_files, csv_file=csv_file, **kws
    )

    track_cells = {}

    df = pd.read_csv(csv_file)
    for index, cell in df.loc[:, "global_cell_index"].items():
        if cell not in track_cells:
            track_cells[cell] = []
            track_cells[cell].append(
                {
                    "init_cell_id": int(df.loc[index, "local_cell_index"]),
                    "init_cell_set": input_cell_set_files[
                        df.loc[index, "local_cellset_index"]
                    ],
                }
            )

    output = {}
    output["track_cells"] = track_cells
    output["input_cell_set_files"] = input_cell_set_files
    output["parameters"] = kws

    with open(json_file, "w") as file:
        json.dump(output, file)

    return True
