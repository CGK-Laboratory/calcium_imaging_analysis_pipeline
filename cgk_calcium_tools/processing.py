import numpy as np
import matplotlib.pyplot as plt
import isx
from typing import Union
import os
import json
import pandas as pd
import unittest

def fix_frames(file: str,
               std_th: Union[None,
                             float] = None,
               report: bool = True) -> int:
    """It fixes inplace a file interpolating its broken frames

    Parameters
    ----------
    file : _type_
        Inscopix movie to edit
    std_th : _type_, optional
        Threshold over the mean fluorescence to detect a broken frame , by default None
    report : bool, optional
        Save a figure reporting the fluorecente metrics in each frame, by default True

    Returns
    -------
    int
        Number of fixed frames
    """
    movie = isx.Movie.read(file)
    num_samples = movie.timing.num_samples
    mean_fluores = np.empty(num_samples)

    for i in range(num_samples):
        frame = movie.get_frame_data(i)
        mean_fluores[i] = frame.mean()

    s_mean_fluores = (mean_fluores - np.mean(mean_fluores)) / \
        np.std(mean_fluores)

    if std_th is not None:
        broken_frames = np.where(s_mean_fluores > std_th)[0]
        dtype = movie.get_frame_data(0).dtype.type
    else:
        broken_frames = np.array([])

    # here additional frames to fix could be added/ommited like dropped or
    # blanks
    indices = np.where(np.diff(broken_frames) > 1)[0] + 1
    consecutive_broken_frames = np.split(broken_frames, indices)
    new_frames = {}
    for cbf in consecutive_broken_frames:
        if len(cbf) == 0:
            continue
        prev_ok_fr = min(cbf) - 1
        next_ok_fr = max(cbf) + 1
        gap = next_ok_fr - prev_ok_fr
        for bf in cbf:
            if prev_ok_fr < 0:
                frame = movie.get_frame_data(next_ok_fr)
            elif next_ok_fr == (num_samples):
                frame = movie.get_frame_data(prev_ok_fr)
            else:
                prev_coef = (bf - prev_ok_fr) / gap
                next_coef = (next_ok_fr - bf) / gap
                frame = next_coef * movie.get_frame_data(next_ok_fr) + \
                    prev_coef * movie.get_frame_data(prev_ok_fr)
            # Frames have to be set in increasing order!
            new_frames[bf] = frame.astype(dtype)

    assert len(broken_frames) <= (0.05 * num_samples), \
        "Too many frames to fix (more than 5%)"

    temp_file = os.path.splitext(file)[0] + '_temp.isxd'
    if len(broken_frames) > 0:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        fixed_movie = isx.Movie.write(
            temp_file, movie.timing, movie.spacing, dtype)

        # fixed_movie_data = np.zeros(list(movie.spacing.num_pixels) + [num_samples],
        #    fixed_movie.data_type)
        for i in range(num_samples):
            if i in new_frames:
                fixed_movie.set_frame_data(i, new_frames[i])
            else:
                fixed_movie.set_frame_data(i, movie.get_frame_data(i))
            # fixed_movie.set_frame_data(i, fixed_movie_data[:, :, i])

        fixed_movie.flush()
        del fixed_movie
        del movie
        os.remove(file)
        os.rename(temp_file, file)
    if report:
        fig = plt.figure()
        plt.plot(s_mean_fluores, c='blue', lw=2)
        for bf in broken_frames:
            plt.axvline(bf, color='red', ls='--', alpha=0.8, lw=1.5)
        if std_th is not None:
            plt.axhline(std_th, color='green', ls='--', alpha=0.8, lw=1.5)

        plt.grid('on')
        plt.ylabel('Standardize Mean Frame Fluorescence')
        plt.xlabel('Frame')
        plt.title('Fixes: {} frames'.format(len(broken_frames)))
        fig.savefig(os.path.splitext(file)[0] + '.png')
        plt.close(fig)
    return len(broken_frames)


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
