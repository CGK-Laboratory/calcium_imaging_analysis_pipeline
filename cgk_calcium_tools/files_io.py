import os
import json
import isx
import numpy as np
from datetime import datetime
from filehandler import isx_files_handler
from pathlib import Path


def create_inscopix_projects(fh: isx_files_handler, cellsetname="pca-ica"):

    src_dir = os.path.dirname(os.path.abspath(__file__))
    with open(
        os.path.join(src_dir, "prj_template.json"), "r", encoding="utf-8"
    ) as file:
        prj_template = file.read()
    with open(
        os.path.join(src_dir, "single_plane_template.json"), "r", encoding="utf-8"
    ) as file:
        single_plane_template = file.read()

    for main_file, single_planes in fh.focus_files.items():
        idxs = [fh.p_rec_paths.index(f) for f in single_planes]
        cellsets = fh.get_results_filenames(f"{cellsetname}", op=None, idx=idxs)
        evs_dets = fh.get_results_filenames(f"{cellsetname}-ED", op=None, idx=idxs)
        dffs = fh.get_results_filenames("dff", op="MC", idx=idxs)

        idx = [fh.rec_paths.index(main_file)]

        project_file = fh.get_results_filenames(
            f"{cellsetname}.isxp", op=None, idx=idx, single_plane=False
        )[0]
        data_folder = project_file[:-4] + "_data"
        os.makedirs(data_folder, exist_ok=True)
        single_planes_info = []
        for dff, cellset, ev_det in zip(dffs, cellsets, evs_dets):
            parsed_plane = single_plane_template
            movie = isx.Movie.read(dff)
            movie_data = movie.get_frame_data(0)
            dmin = np.min(movie_data)
            dmax = np.max(movie_data)
            del movie
            replacements = {
                "{eventdet_path}": ev_det,
                "{eventdet_name}": Path(ev_det).name,
                "{cellset_path}": cellset,
                "{cellset_name}":  Path(cellset).name,
                "{DFF_path}": dff,
                "{DFF_name}": Path(dff).name,
                "{dmax}": str(dmax),
                "{dmin}": str(dmin)
            }
            
            for key, value in replacements.items():
                parsed_plane = parsed_plane.replace(key, value)

            single_planes_info.append(parsed_plane)

        prj_text = prj_template.replace(
            "{prj_name}", Path(project_file).name
        ).replace("{plane_1ist}", ", ".join(single_planes_info))
        with open(project_file, "wt") as file:
            file.write(prj_text)


def same_json_or_remove(
    parameters: dict, input_files_keys: list, output: str, verbose: bool
) -> bool:
    """
    If the file exist and the json is the same returns True.
    Else: removes the file and the json associated with it.

    Parameters
    ----------
     parameters : dict
        Parameter dictionary of executed functions.
    input_files_keys : list
        list wuth the parameters keys
    output : str
        output files path
    verbose : bool, optional
        Show additional messages, by default False

    Returns
    -------
    bool
        Returns True if the file and json exists. Else returns False

    """
    json_file = json_filename(output)
    if os.path.exists(json_file):
        if os.path.exists(output):
            if check_same_existing_json(
                parameters, json_file, input_files_keys, verbose
            ):
                if verbose:
                    print(f"File {output} already created with these parameters")
                return True
        os.remove(json_file)
    if os.path.exists(output):
        os.remove(output)
    return False


def check_same_existing_json(
    parameters: dict, json_file: str, input_files_keys: list, verbose: bool
) -> bool:
    """
    Go through the new parameters checking for diferences.
    Missing parameters in new parameters omitted

    Parameters
    ----------
    parameters : dict
        Parameter dictionary of executed functions.
    json_file : str
       json files path
    input_files_keys : list
        list with the parameters keys
    verbose : bool, optional
        Show additional messages, by default False

    Returns
    -------
    bool
        Returns True if there are no diferences in the parameters with the json file. Else returns False

    """
    with open(json_file) as file:
        prev_parameters = json.load(file)
    for key, value in parameters.items():
        # only comments can be different
        if key != "comments":
            if key not in prev_parameters:
                if verbose:
                    print(f"new parameter {key}")
                return False
            elif prev_parameters[key] != value:
                if key == "isx_version":
                    if verbose:
                        print(
                            f"Warning. file created with isx version {prev_parameters[key]}, current:{value}"
                        )
                    continue
                if verbose:
                    print(f"different {key}: old:{prev_parameters[key]}, new:{value}")
                return False

    # Check dates for all input files dates are consistent
    for input_file_key in input_files_keys:
        input_files = parameters[input_file_key]

        if isinstance(input_files, str):
            # generalize for input fields containing lists
            input_files = [input_files]
        base_path = os.path.dirname(
            json_file
        )  # to get the absolut path from json file and add it to input_file, which is a relative path
        for input_file in input_files:
            json_file = json_filename(os.path.join(base_path, input_file))
            if os.path.exists(json_file):
                with open(json_file) as in_file:
                    in_data = json.load(in_file)
                if prev_parameters["input_modification_date"] < in_data["date"]:
                    old_date = prev_parameters["input_modification_date"]
                    new_date = in_data["date"]
                    if verbose:
                        print(
                            f"updated file {json_file}: old:{old_date}, new:{new_date}"
                        )
                    return False
    return True


def json_filename(filename: str) -> str:
    """
    gives the json path asociated to the original file path

    Parameters
    ----------
    filename : str
       file path

    Returns
    -------
    str
        Returns the json path as string

    """
    return os.path.splitext(filename)[0] + ".json"


def remove_file_and_json(output: str) -> None:
    """
    Removes the original file path and the asociated json file

    Parameters
    ----------
    output : str
       file path

    Returns
    -------
    None

    """
    if os.path.exists(output):
        os.remove(output)
    json_file = json_filename(output)
    if os.path.exists(json_file):
        os.remove(json_file)


def write_log_file(
    params: dict,
    dir_name: str,
    extra_params: dict = {},
    input_files_keys: list = ["input_movie_files"],
    output_file_key: str = "output_movie_files",
) -> None:
    """
    Removes the original file path and the asociated json file

    Parameters
    ----------
    params : dict
        Parameter dictionary of executed functions. Paths should be relative.
    extra_params : dict, optional
       Parameter dictionary, by default empty.
    input_files_keys : list, optional
        list with the parameters keys, by default ["input_movie_files"]
    output_file_key :  str, optional
        key to access output file, by default "output_movie_files".
    dirname : str
        Combine the dirname with the basename to obtain the absolute path.
    Returns
    -------
    None

    """

    data = params.copy()
    data.update(extra_params)
    data["isx_version"] = isx.__version__
    if not isinstance(data[output_file_key], list):
        data_output_file = [data[output_file_key]]
    else:
        data_output_file = data[output_file_key]
    for output in data_output_file:
        log_path = json_filename(output)
        actual_date = datetime.utcnow()
        data["input_modification_date"] = None
        if not isinstance(input_files_keys, list):
            input_files_keys = [input_files_keys]
        temp_date_str = ""
        for input_file_key in input_files_keys:
            input_files = data[input_file_key]
            if not isinstance(input_files, list):
                input_files = [input_files]
            for input_file in input_files:
                input_json = json_filename(input_file)
                if os.path.exists(os.path.join(dir_name, input_json)):
                    with open(os.path.join(dir_name, input_json)) as file:
                        input_data = json.load(file)
                    temp_date_str = max(input_data["date"], temp_date_str)
        data["input_modification_date"] = temp_date_str
        data["date"] = actual_date.strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(dir_name, log_path), "w") as file:
            json.dump(data, file, indent=4)
