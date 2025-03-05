
from .isx_aux_functions import  create_similar_empty_cellset
import isx
from typing import Union
from .processing import fix_frames
from .files_io import (
    write_log_file,
    same_json_or_remove,
    RecordingFile
)
import os
import numpy as np
from .pipeline_functions import register


@register('isx:preprocessing',"Applying Bandpass Filter...", 'PP')
def isx_preprocessing(recording,output_folder, suffix, parameters, verbose) -> None:
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
    file_fullpath = os.path.join(recording.main_folder,  recording.file)
    fname,ext = os.path.splitext(recording.file)
    out_file_rel = fname + f"-{suffix}." + ext
    out_file = os.path.join(output_folder,out_file_rel)

    log_parameters={
                'function_params':
                {       
                    'source_rel': recording.file,
                    'output_rel':out_file_rel,
                    **parameters
                },             
                'isx_version':isx.__version__,
                'input_files':[recording.file]
            }
    # fix_frames_th_std is an additional parameter that is not used by isx.preprocess
    fix_frames_th_std = parameters.pop("fix_frames_th_std") 

    if os.path.exists(out_file):
        if same_json_or_remove(
            parameters = log_parameters,
            output=out_file,
            verbose=False):
                return out_file, None

    isx.preprocess(input_movie_files=file_fullpath,
                    output_movie_files=out_file,**parameters
    )

    nfixed,output_figure = fix_frames(out_file, std_th=fix_frames_th_std, report=True)

    write_log_file(
        log_parameters,
        file_outputs=[out_file_rel],
        dir_name=output_folder
    )
    if verbose:
        print("{} preprocessing completed. {} frames fixed.".format(out_file_rel, nfixed))
    return output_figure



#@register('isx:cnmfe',"Extracting Cells...",'cnmfe')
@register('isx:pca_ica',"Extracting Cells...",'pca_ica')
def extract_cells( #broken
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
                    json_file = os.path.join(os.path.dirname(fout), os.path.splitext(fout)[0], ".json")
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
                **parameters_for_isx(
                    parameters,
                    {"input_movie_files": input, "output_cell_set_files": output},
                )
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

@register('isx:de_interleave',"De-interleaving multiplane movies...",'DI')
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
        fname,ext = os.path.splitext(recording.file)
        sp_files_rel = [fname + f"-{suffix}_{focus}." + ext for focus in recording.focus]
        sp_files = [os.path.join(output_folder,f) for f in sp_files_rel]
        for sp_file in sp_files_rel:
           
            parameters={
                        'function_params':{
                            "source_rel": main_file_rel,
                            "output_rel": sp_files_rel,
                            "efocus": recording.focus
                        },                        
                        'isx_version':isx.__version__,
                        'input_files':[main_file_rel]
                    }

            if os.path.exists(sp_file):
                if same_json_or_remove(
                    parameters = parameters,
                    output=sp_file,
                    verbose=False,
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
            params=parameters,
            dir_name=output_folder,
            file_outputs=sp_files_rel,
        )

    return sp_files, None

@register('isx:trim_movie',"Trimming",'TR')
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
            parameters,
            ["comments", "video_len"],
            {"input_movie_file": input, "output_movie_file": output},
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
    

@register('isx:spatial_filter',"Preprocessing",'BP')
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
            parameters,
            ["comments"],
            {"input_movie_files": input, "output_movie_files": output}
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



@register('isx:motion_correct',"Applying Motion Correction...",'MC')
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


@register('isx:dff',"Normalizing via DF/F0",'DFF')
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


@register('isx:project_movie',"Projecting Movies",'PM')
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
