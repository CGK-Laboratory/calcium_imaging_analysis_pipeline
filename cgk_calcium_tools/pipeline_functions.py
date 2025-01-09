import isx
from typing import Union,  Tuple
from .processing import fix_frames
from .isx_aux_functions import create_empty_events, cellset_is_empty, create_empty_cellset
from typing import Callable
from .files_io import (
    write_log_file,
    same_json_or_remove,
    parameters_for_isx,
)
import os
import numpy as np
import shutil
f_register: dict[str, Callable] = {}
f_message: dict[str, str] = {}
def register(name: Union[str, list], message=None) -> None:
    
    if message is None:
        message = f"Running function: {name}"

    def decorator(func: Callable) -> None:
        assert isinstance(func, Callable)
        if  isinstance(name, list):            
            for n in name:
                f_register[n] = func
                f_message[n] = message
        else:
            f_register[name] = func
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


@register(['isx:project_movie', "project_movie"],"Projecting Movies")
def project_movie(input, output, parameters: dict, verbose:bool=False) -> None:
    """
    This function applies isx.project_movie to project movies to a single statistic image.

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
        "output_image_file": os.path.basename(output),
    }
    parameters.update(new_data)
    if same_json_or_remove(
        parameters,
        input_files_keys=["input_movie_files"],
        output=output,
        verbose=verbose,
    ):
        return
    isx.project_movie(
        **parameters_for_isx(
            parameters,
            ["comments"],
            {"input_movie_files": input, "output_image_file": output},
        )
    )
    write_log_file(
        parameters,
        os.path.dirname(output),
        {"function": "project_movie"},
        input_files_keys=["input_movie_files"],
        output_file_key="output_image_file",
    )


@register(['isx:trim_movie','trim'],"Trimming")
def trim_movie(
    input, output,
    user_parameters: dict,
    verbose: bool = False,
) -> None:
    """
    it invokes the isx.trim_movie function, which trims frames from a movie to generate a new movie

    Parameters
    ----------
    verbose : bool, optional
        Show additional messages, by default False

    Returns
    -------
    None

    """

    assert (
        user_parameters["video_len"] is not None
    ), "Trim movie requires parameter video len"


    parameters = {
        "input_movie_file": os.path.basename(input),
        "output_movie_file": os.path.basename(output),
    }
    movie = isx.Movie.read(input)
    sr = 1 / (movie.timing.period.to_msecs() / 1000)
    endframe = user_parameters["video_len"] * sr
    maxfileframe = movie.timing.num_samples + 1
    assert maxfileframe >= endframe, "max time > duration of the video"
    parameters["video_len"] = user_parameters["video_len"]
    if same_json_or_remove(
        parameters,
        input_files_keys=["input_movie_file"],
        output=output,
        verbose=verbose,
    ):
        return
    parameters["crop_segments"] = [[endframe, maxfileframe]]
    isx.trim_movie(
        **parameters_for_isx(
            parameters,
            ["comments", "video_len"],
            {"input_movie_file": input, "output_movie_file": output},
        )
    )
    if verbose:
        print("{} trimming completed".format(output))
    del parameters["crop_segments"]
    write_log_file(
        parameters,
        os.path.dirname(output),
        {"function": "trimming"},
        input_files_keys=["input_movie_file"],
        output_file_key="output_movie_file",
    )

@register(['isx:event_detection','event_detection'],"Detecting Events")
def isx_event_detection(input, output, ed_parameters: dict, verbose:bool=False) -> None:
    new_data = {
            "input_cell_set_files": os.path.basename(input),
            "output_event_set_files": os.path.basename(output),
    }
    ed_parameters.update(new_data)
    if same_json_or_remove(
        ed_parameters,
        output=output,
        verbose=verbose,
        input_files_keys=["input_cell_set_files"],
    ):
        return
    try:
        isx.event_detection(
            **parameters_for_isx(
                ed_parameters,
                ["comments"],
                {
                    "input_cell_set_files": input,
                    "output_event_set_files": output,
                },
            )
        )
    except Exception as e:
        print(
            f"Warning: Event_detection, failed to create file: {output}.\n"
            + "Empty file created with its place"
        )
        create_empty_events(input, output)
    write_log_file(
        ed_parameters,
        os.path.dirname(output),
        {"function": "event_detection"},
        input_files_keys=["input_cell_set_files"],
        output_file_key="output_event_set_files",
    )


@register(['isx:auto_accept_reject','auto_accept_reject'],"Accepting/Rejecting Cells")
def isx_auto_accept_reject(input_cell_set, input_event_set, ar_cell_set_file, ar_parameters: dict, verbose:bool=False) -> None:
    new_data = {
        "input_cell_set_files": os.path.basename(input_cell_set),
        "input_event_set_files": os.path.basename(input_event_set),
    }
    ar_parameters.update(new_data)
    try:
        isx.auto_accept_reject(
            **parameters_for_isx(
                ar_parameters,
                ["comments"],
                {
                    "input_cell_set_files": input_cell_set,
                    "input_event_set_files": input_event_set,
                },
            )
        )
    except Exception as e:
        if verbose:
            print(e)
    write_log_file(
        ar_parameters,
        os.path.dirname(input_cell_set),
        {"function": "accept_reject", "config_json": ar_cell_set_file},
        input_files_keys=["input_cell_set_files", "input_event_set_files"],
        output_file_key="config_json",
    )

@register(['isx:deconvolve_cellset','deconvolve_cellset'],"Running Deconvolution Registration")
def isx_deconvolve_cellset(cellset,denoise_file,ed_file, parameters: dict, verbose:bool=False) -> None:

    new_data = {
        "input_raw_cellset_files": os.path.basename(cellset),
        "output_denoised_cellset_files": os.path.basename(denoise_file),
        "output_spike_eventset_files": os.path.basename(ed_file),
    }
    parameters.update(new_data)

    if not same_json_or_remove(
        parameters,
        output=denoise_file,
        verbose=verbose,
        input_files_keys=["input_raw_cellset_files"],
    ):
        if cellset_is_empty(cellset, accepted_only=parameters["accepted_only"]):
            create_empty_events(cellset, ed_file)
            create_empty_cellset(cellset, denoise_file)
        else:
            isx.deconvolve_cellset(
                **parameters_for_isx(
                    parameters,
                    ["comments"],
                    {
                        "input_raw_cellset_files": cellset,
                        "output_denoised_cellset_files": denoise_file,
                        "output_spike_eventset_files": ed_file,
                    },
                )
            )

        for ofile, outkey in (
            (denoise_file, "output_denoised_cellset_files"),
            (ed_file, "output_spike_eventset_files"),
        ):
            write_log_file(
                parameters,
                os.path.dirname(ofile),
                {"function": "deconvolve_cellset"},
                input_files_keys=["input_raw_cellset_files"],
                output_file_key=outkey,
            )

@register(['isx:multiplane_registration','multiplane_registration'],"Multiplane Registration...")
def isx_multiplane_registration(input_cell_set_files, ar_cell_set_file, output_cell_set_file, mpr_parameters: dict, verbose:bool=False) -> None:
    input_cell_set_file_names = [
        os.path.basename(file) for file in input_cell_set_files
    ]
    new_data = {
        "input_cell_set_files": input_cell_set_file_names,
        "output_cell_set_file": os.path.basename(output_cell_set_file),
        "auto_accept_reject": os.path.basename(ar_cell_set_file),
                }
    mpr_parameters.update(new_data)

    if same_json_or_remove(
        mpr_parameters,
        output=output_cell_set_file,
        verbose=verbose,
        input_files_keys=["input_cell_set_files", "auto_accept_reject"],
    ):
        return
    input_cellsets = []
    for i in input_cell_set_files:
        if not cellset_is_empty(i):
            input_cellsets.append(i)

    if len(input_cellsets) == 0:
        print(
            f"Warning: File: {output_cell_set_file} not generated.\n"
            + "Empty cellmap created in its place"
        )
        create_empty_cellset(
            input_file=input_cell_set_files[0],
            output_cell_set_file=output_cell_set_file,
        )

    elif len(input_cellsets) == 1:
        shutil.copyfile(input_cellsets[0], output_cell_set_file)
    else:
        isx.multiplane_registration(
            **parameters_for_isx(
                mpr_parameters,
                ["comments", "auto_accept_reject"],
                {
                    "input_cell_set_files": input_cellsets,
                    "output_cell_set_file": output_cell_set_file,
                },
            )
        )

    write_log_file(
        mpr_parameters,
        os.path.dirname(output_cell_set_file),
        {"function": "multiplane_registration"},
        input_files_keys=["input_cell_set_files", "auto_accept_reject"],
        output_file_key="output_cell_set_file",
    )

def de_interleave(main_file, planes_fs, focus, overwrite: bool = False) -> None:
    """
    This function applies the isx.de_interleave function, which de-interleaves multiplane movies

    Parameters
    ----------
    overwrite : Bool, optional
        If True the function erases the previous information; otherwise, it appends to a list.
        By default "False"

    Returns
    -------
    None

    """

    # Initialize progress bar
    if len(focus) > 1:  # has multiplane
        existing_files = []
        for sp_file in planes_fs:
            dirname = os.path.dirname(sp_file)
            json_file = os.path.splitext(sp_file)[0] + ".json"
            if os.path.exists(sp_file):
                if overwrite:
                    os.remove(sp_file)
                    if os.path.exists(json_file):
                        os.remove(json_file)
                else:
                    if same_json_or_remove(
                        parameters={
                            "input_movie_files": main_file,
                            "output_movie_files": [
                                os.path.basename(p) for p in planes_fs
                            ],
                            "in_efocus_values": focus,
                        },
                        input_files_keys=["input_movie_files"],
                        output=sp_file,
                        verbose=False,
                    ):
                        existing_files.append(sp_file)
        if len(existing_files) != len(planes_fs):  # has files to run
            for f in existing_files:  # remove existing planes
                os.remove(f)
            try:
                isx.de_interleave(main_file, planes_fs, focus)

            except Exception as err:
                print("Reading: ", main_file)
                print("Writting: ", planes_fs)
                raise err

        data = {
            "input_movie_files": main_file,
            "output_movie_files": [os.path.basename(p) for p in planes_fs],
            "in_efocus_values": focus,
        }
        # for sp_file in planes_fs:
        write_log_file(
            params=data,
            dir_name=dirname,
            input_files_keys=["input_movie_files"],
            output_file_key="output_movie_files",
        )

    return

