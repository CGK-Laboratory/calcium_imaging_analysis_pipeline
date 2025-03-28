from .isx_aux_functions import create_similar_empty_cellset
import isx
from typing import Union
from .processing import fix_frames
from .files_io import write_log_file, same_json_or_remove, RecordingFile
import os
import numpy as np
from .pipeline_functions import register


@register("isx:preprocessing++", "Applying Bandpass Filter...", "PP")
def isx_preprocessing(recording, output_folder, suffix, parameters, verbose) -> None:
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
    file_fullpath = os.path.join(recording.main_folder, recording.file)
    fname, ext = os.path.splitext(recording.file)
    out_file_rel = fname + f"-{suffix}." + ext
    out_file = os.path.join(output_folder, out_file_rel)

    log_info = {
        "function_params": parameters.copy(),
        "isx_version": isx.__version__,
        "function": "isx:preprocessing++",
        "input_files": [recording.file],
    }
    # fix_frames_th_std is an additional parameter that is not used by isx.preprocess
    fix_frames_th_std = parameters.pop("fix_frames_th_std")

    if os.path.exists(out_file):
        if same_json_or_remove(log_info=log_info, output=out_file, verbose=False):
            return out_file, None

    isx.preprocess(
        input_movie_files=file_fullpath, output_movie_files=out_file, **parameters
    )

    nfixed, output_figure = fix_frames(out_file, std_th=fix_frames_th_std, report=True)

    write_log_file(log_info, file_outputs=[out_file_rel], dir_name=output_folder)
    if verbose:
        print(
            "{} preprocessing completed. {} frames fixed.".format(out_file_rel, nfixed)
        )
    return output_figure


# @register('isx:cnmfe',"Extracting Cells...",'cnmfe')
@register("isx:pca_ica", "Extracting Cells...", "pca_ica")
def extract_cells(  # broken
    self,
    alg: str,
    overwrite: bool = False,
    verbose: bool = False,
    cells_extr_params: Union[dict, None] = None,
    cellsetname: Union[str, None] = None,
) -> None:
    """
    This function run a cell extraction algorithm, detect events
    and auto accept_reject

    Parameters
    ----------
    alg : str
        Cell extraction algorithm: 'cnmfe' or 'pca-ica'
    overwrite : bool, optional
        Force compute everything again, by default False
    verbose : bool, optional
        Show additional messages, by default False
    cells_extr_params : Union[dict,None], optional
        Parameters for cell extraction, by default None
    detection_params : Union[dict,None], optional
        Parameters for event detection, by default None
    accept_reject_params : Union[dict,None], optional
        Parameters for automatic accept_reject cell, by default None
    Returns
    -------
    None

    """
    assert alg in ["pca-ica", "cnmfe"], "alg must be 'pca-ica' or 'cnmfe'."

    # extract cells
    parameters = self.default_parameters[alg].copy()
    if cells_extr_params is not None:
        for key, value in cells_extr_params.items():
            assert key in parameters, f"The parameter: {key} does not exist"
            parameters[key] = value

    if alg == "pca-ica":
        cell_det_fn = isx.pca_ica
        # pca uses dff:
        inputs_files = self.get_results_filenames("dff", op="MC")
    elif alg == "cnmfe":
        cell_det_fn = isx.run_cnmfe
        inputs_files = self.get_filenames(op=None)
    else:
        raise "alg must be 'pca-ica' or 'cnmfe'."
    if cellsetname is None:
        cellsetname = alg
    cellsets = self.get_results_filenames(f"{cellsetname}", op=None)

    if overwrite:
        for fout in cellsets:
            if os.path.exists(fout):
                os.remove(fout)
                json_file = os.path.join(
                    os.path.dirname(fout), os.path.splitext(fout)[0], ".json"
                )
                if os.path.exists(json_file):
                    os.remove(json_file)

    for input, output in zip(inputs_files, cellsets):
        new_data = {
            "input_movie_files": os.path.basename(input),
            "output_cell_set_files": os.path.basename(output),
        }
        parameters.update(new_data)
        if same_json_or_remove(
            parameters,
            input_files_keys=["input_movie_files"],
            output=output,
            verbose=verbose,
        ):
            continue
        cell_det_fn(
            parameters,
            {"input_movie_files": input, "output_cell_set_files": output},
        )

        if not os.path.exists(output):
            print(
                f"Warning: Algorithm {alg}, failed to create file: {output}.\n"
                + "Empty cellmap created with its place"
            )
            create_similar_empty_cellset(
                input_file=input,
                output_cell_set_file=output,
            )

        write_log_file(
            parameters,
            os.path.dirname(output),
            {"function": alg},
            input_files_keys=["input_movie_files"],
            output_file_key="output_cell_set_files",
        )
    if verbose:
        print("Cell extraction, done")


@register("isx:de_interleave", "De-interleaving multiplane movies...", "DI")
def de_interleave(recording, output_folder, suffix, verbose: bool, **kws) -> None:
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
    main_file_rel = recording.file
    main_file = os.path.join(recording.main_folder, main_file_rel)
    # Initialize progress bar
    if len(recording.focus) > 1:  # has multiplane
        existing_files = []
        fname, ext = os.path.splitext(recording.file)
        sp_files_rel = [
            fname + f"-{suffix}_{focus}." + ext for focus in recording.focus
        ]
        sp_files = [os.path.join(output_folder, f) for f in sp_files_rel]
        for sp_file in sp_files_rel:

            log_info = {
                "function_params": {},
                "function": "isx:de_interleave",
                "isx_version": isx.__version__,
                "input_files": [main_file_rel],
            }

            if os.path.exists(sp_file):
                if same_json_or_remove(
                    log_info=log_info,
                    output=sp_file,
                    verbose=verbose,
                ):
                    existing_files.append(sp_file)

        if len(existing_files) != len(recording.file):  # some file is missing
            for f in existing_files:  # remove existing planes
                os.remove(f)
                os.remove(os.path.splitext(f)[0] + ".json")
            try:
                isx.de_interleave(main_file, sp_files, recording.fileocus)

            except Exception as err:
                print("Reading: ", main_file)
                print("Writting: ", sp_files)
                raise err

        # for sp_file in planes_fs:
        write_log_file(
            log_info=log_info,
            dir_name=output_folder,
            file_outputs=sp_files_rel,
        )

    return sp_files, None


@register("isx:trim_movie++", "Trimming", "TR")
def trim_movie(recording, output_folder, suffix, verbose: bool, **parameters) -> None:
    """
    Run isx.trim_movie function to extract a video segment with parameters in ms from the start.
    """

    input_file = os.path.join(recording.main_folder, recording.file)
    fname, ext = os.path.splitext(recording.file)
    output_file_rel = fname + f"-{suffix}." + ext
    output_file = os.path.join(output_folder, output_file_rel)

    log_info = {
        "function_params": parameters.copy(),
        "function": "isx:motion_correct",
        "isx_version": isx.__version__,
        "input_files": [recording.file],
    }

    movie = isx.Movie.read(input)
    sr_khz = 1 / (movie.timing.period.to_msecs())
    start_frame = 1 + int(parameters["video_start_ms"] * sr_khz)
    end_frame = int(start_frame + parameters["video_len_m"] * sr_khz)

    maxfileframe = movie.timing.num_samples + 1
    del movie
    assert maxfileframe >= end_frame, "max time > duration of the video"

    if same_json_or_remove(
        log_info,
        output=output_file,
        verbose=verbose,
    ):
        return
    if start_frame > 1:
        parameters["crop_segments"] = [[1, start_frame - 1]]
    else:
        parameters["crop_segments"] = []
    if end_frame < maxfileframe:
        parameters["crop_segments"].append([end_frame + 1, maxfileframe])

    parameters = {
        "input_movie_file": input_file,
        "output_movie_file": output_file,
    }
    isx.trim_movie(
        **parameters,
    )
    if verbose:
        print("{} trimming completed".format(output_file_rel))
    write_log_file(
        log_info=log_info, dir_name=output_folder, file_outputs=[output_file_rel]
    )


@register("isx:spatial_filter", "Preprocessing", "BP")
def isx_spatial_filter(
    recording, output_folder: str, suffix: str, verbose: bool, **parameters
) -> None:
    """
    Run isx.spatial_filter
    """

    input_file_rel = recording.file
    input_file = os.path.join(recording.main_folder, input_file_rel)
    fname, ext = os.path.splitext(recording.file)
    output_file_rel = fname + f"-{suffix}." + ext
    output_file = os.path.join(output_folder, output_file_rel)

    log_info = {
        "function_params": parameters.copy(),
        "function": "isx:spatial_filter",
        "isx_version": isx.__version__,
        "input_files": [input_file_rel],
    }

    parameters.update(
        {"input_movie_files": [input_file], "output_movie_files": [output_file]}
    )
    if same_json_or_remove(
        log_info,
        output=output_file,
        verbose=verbose,
    ):
        return

    isx.spatial_filter(**parameters)
    write_log_file(
        log_info=log_info,
        dir_name=output_folder,
        file_outputs=[output_file_rel],
    )
    if verbose:
        print("{} bandpass filtering completed".format(output_file_rel))


@register("isx:motion_correct", "Applying Motion Correction...", "MC")
def isx_motion_correct(
    recording, output_folder, suffix, verbose: bool, **parameters
) -> None:
    """
    Run isx.motion_correct, which motion correct movies to a reference frame.
    """
    input_file = os.path.join(recording.main_folder, recording.file)
    fname, ext = os.path.splitext(recording.file)
    output_file_rel = fname + f"-{suffix}." + ext
    output_file = os.path.join(output_folder, output_file_rel)

    simple_name = os.path.splitext(output_file_rel)[0]
    translation_file = os.path.join(output_folder, simple_name + "-translations.csv")
    crop_rect_file = os.path.join(output_folder, simple_name + "-crop_rect.csv")

    log_info = {
        "function_params": parameters.copy(),
        "function": "isx:motion_correct",
        "isx_version": isx.__version__,
        "input_files": [recording.file],
    }

    if same_json_or_remove(
        log_info,
        output=output_file,
        verbose=verbose,
    ):
        return

    parameters.update(
        {
            "input_movie_files": [input_file],
            "output_movie_files": [output_file],
            "output_translation_files": translation_file,
            "output_crop_rect_file": crop_rect_file,
        }
    )
    isx.motion_correct(**parameters)
    write_log_file(
        log_info=log_info, dir_name=output_folder, file_outputs=[output_file_rel]
    )

    if verbose:
        print("{} motion correction completed".format(output_file_rel))


@register("isx:dff", "Normalizing via DF/F0", "DFF")
def isx_dff(recording, output_folder, suffix, verbose: bool, **parameters) -> None:
    """
    Run isx.dff function, to compute the DF/F movies.
    """
    input_file = os.path.join(recording.main_folder, recording.file)
    fname, ext = os.path.splitext(recording.file)
    output_file_rel = fname + f"-{suffix}." + ext
    output_file = os.path.join(output_folder, output_file_rel)

    log_info = {
        "function_params": parameters.copy(),
        "function": "isx:dff",
        "isx_version": isx.__version__,
        "input_files": [recording.file],
    }

    if same_json_or_remove(
        log_info,
        output=output_file,
        verbose=verbose,
    ):
        return

    parameters.update(
        {
            "input_movie_files": [input_file],
            "output_movie_files": [output_file],
        }
    )

    isx.dff(
        **parameters,
    )
    write_log_file(
        log_info=log_info, dir_name=output_folder, file_outputs=[output_file_rel]
    )


@register("isx:project_movie", "Projecting Movies", "PM")
def project_movie(
    recording, output_folder, suffix, verbose: bool, **parameters
) -> None:
    """
    Run isx.project_movie to project movies to a single statistic image.
    """
    input_file = os.path.join(recording.main_folder, recording.file)
    fname, ext = os.path.splitext(recording.file)
    output_file_rel = fname + f"-{suffix}." + ext
    output_file = os.path.join(output_folder, output_file_rel)

    log_info = {
        "function_params": parameters.copy(),
        "function": "isx:project_movie",
        "isx_version": isx.__version__,
        "input_files": [recording.file],
    }

    if same_json_or_remove(
        log_info,
        output=output_file,
        verbose=verbose,
    ):
        return
    parameters.update(
        {
            "input_movie_files": [input_file],
            "output_image_file": [output_file],
        }
    )
    isx.project_movie(**parameters)
    write_log_file(
        log_info=log_info, dir_name=output_folder, file_outputs=[output_file_rel]
    )
