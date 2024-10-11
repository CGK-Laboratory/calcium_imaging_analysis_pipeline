import isx
from typing import Union,  Tuple
from .processing import fix_frames
from typing import Callable
from .files_io import (
    write_log_file,
    same_json_or_remove,
    parameters_for_isx,
)
import os
import numpy as np
f_register: dict[str, Callable] = {}
f_message: dict[str, str] = {}
def register(name: Union[str, list],message=None) -> None:
    
    if message is None:
        message = f"Running function: {name}"

    def decorator(func: Callable) -> None:
        assert isinstance(func, Callable)
        if not isinstance(name, list):
            name = [name]
        for n in name:
            f_register[n] = func
            f_message[name] = message
    return decorator

        
@register(['isx:spatial_filter', "spatial_filter"],"Preprocessing")
def isx_spatial_filter(input, output, parameters, verbose) -> None:
    """
    Applies spatial bandpass filtering to each frame of one or more movies

    Parameters
    ----------
    parameters : dict
        Parameter dictionary of executed functions.
    verbose : bool, optional
        Show additional messages, by default False
    Returns
    -------
    None

    """

    parameters_rel = parameters.copy()
    parameters_rel.update(
        {
            "input_movie_files": os.path.basename(parameters["input_movie_files"]),
            "output_movie_files": os.path.basename(parameters["output_movie_files"])
        }
    )


    if same_json_or_remove(
        parameters_rel,
        input_files_keys=["input_movie_files"],
        output=parameters["output_movie_files"],
        verbose=verbose,
    ):
        return
    isx.spatial_filter(
        **parameters_for_isx(
            parameters,
            ["comments"],
            {"input_movie_files": input, "output_movie_files": output},
        )
    )
    write_log_file(
        parameters_rel,
        os.path.dirname(output),
        {"function": "spatial_filter"},
        input_files_keys=["input_movie_files"],
        output_file_key="output_movie_files",
    )
    if verbose:
        print("{} bandpass filtering completed".format(output))



@register(['isx:motion_correct', "motion_correct"],"Applying Motion Correction to")
def isx_motion_correct(input, output, parameters, verbose) -> None:
    """
    After checks, use the isx.motion_correct function, which motion correct movies to a reference frame.

    Parameters
    ----------
    parameters : dict
       Parameter list of executed functions.
    verbose : bool, optional
        Show additional messages, by default False

    Returns
    -------
    None

    Examples
    --------
    """
    # Initialize progress bar

    translation_file = os.path.splitext(output)[0] + "-translations.csv"
    crop_rect_file = os.path.splitext(output)[0] + "-crop_rect.csv"

    new_data = {
        "input_movie_files": os.path.basename(input),
        "output_movie_files": os.path.basename(output),
        "output_translation_files": os.path.basename(translation_file),
        "output_crop_rect_file": os.path.basename(crop_rect_file),
    }
    parameters.update(new_data)
    if same_json_or_remove(
        parameters,
        input_files_keys=["input_movie_files"],
        output=output,
        verbose=verbose,
    ):
        return
    isx.motion_correct(
        **parameters_for_isx(
            parameters,
            ["comments"],
            {
                "input_movie_files": input,
                "output_movie_files": output,
                "output_translation_files": translation_file,
                "output_crop_rect_file": crop_rect_file,
            },
        )
    )
    write_log_file(
        parameters,
        os.path.dirname(output),
        {"function": "motion_correct"},
        input_files_keys=["input_movie_files"],
        output_file_key="output_movie_files",
    )

    if verbose:
        print("{} motion correction completed".format(output))


@register(['isx:preprocessing', "preprocessing"],"Applying Bandpass Filter to")
def isx_preprocessing(input, output, parameters, verbose) -> None:
    """
    After performing checks, use the isx.preprocess function, which preprocesses movies,
    optionally applying spatial and temporal downsampling and cropping.

    Parameters
    ----------

    parameters : dict
       Parameter dictionary of executed functions.
    verbose : bool, optional
        Show additional messages, by default False

    Returns
    -------
    None

    """
    if isinstance(parameters["spatial_downsample_factor"], str):
        res_idx, value = parameters["spatial_downsample_factor"].split("_")
        if res_idx == "maxHeight":
            idx_resolution = 0
        elif res_idx == "maxWidth":
            idx_resolution = 1
        else:
            assert False, "error in sp_downsampling parameter value"
        movie = isx.Movie.read(input)
        resolution = movie.spacing.num_pixels
        del movie
        parameters["spatial_downsample_factor"] = np.ceil(
            resolution[idx_resolution] / float(value)
        )

    parameters.update(
        {
            "input_movie_files": os.path.basename(input),
            "output_movie_files": os.path.basename(output),
        }
    )

    if same_json_or_remove(
        parameters,
        input_files_keys=["input_movie_files"],
        output=output,
        verbose=verbose,
    ):
        return

    isx.preprocess(
        **parameters_for_isx(
            parameters,
            ["comments", "fix_frames_th_std"],
            {"input_movie_files": input, "output_movie_files": output},
        )
    )

    nfixed = fix_frames(output, std_th=parameters["fix_frames_th_std"], report=True)

    write_log_file(
        parameters,
        os.path.dirname(output),
        {"function": "preprocess"},
        input_files_keys=["input_movie_files"],
        output_file_key="output_movie_files",
    )
    if verbose:
        print("{} preprocessing completed. {} frames fixed.".format(output, nfixed))



@register(['isx:dff', "dff"],"Normalizing via DF/F0")
def isx_dff(input, output, parameters, verbose) -> None:

    """
    This function applies isx.dff function, to compute the DF/F movies.

    Parameters
    ----------
    overwrite : bool, optional
        Remove results and recompute them, by default False
    verbose : bool, optional
        Show additional messages, by default False

    Returns
    -------
    None

    """
    
    new_data = {
        "input_movie_files": os.path.basename(input),
        "output_movie_files": os.path.basename(output),
    }
    parameters.update(new_data)
    if same_json_or_remove(
        parameters,
        input_files_keys=["input_movie_files"],
        output=output,
        verbose=verbose,
    ):
        return
    isx.dff(
        **parameters_for_isx(
            parameters,
            ["comments"],
            {"input_movie_files": input, "output_movie_files": output},
        )
    )
    write_log_file(
        parameters,
        os.path.dirname(output),
        {"function": "dff"},
        input_files_keys=["input_movie_files"],
        output_file_key="output_movie_files",
    )

