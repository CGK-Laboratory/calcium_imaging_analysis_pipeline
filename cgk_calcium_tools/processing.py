import numpy as np
import matplotlib.pyplot as plt
import isx
from typing import Union
import os


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
