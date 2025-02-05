import os
import json
from datetime import datetime
from typing import NamedTuple, Iterable, Union


class RecordingFile(NamedTuple):
    file: str = ""
    main_folder: str = ""
    efocus: Iterable[Union[int, float]]
    resolution: Iterable = []
    duration: Union[int, float]
    frames_per_second: Union[int, float]
    additional_files: list = []
    update_timestamp: str = ""
    creation_function: str = ""
    source_files: list = []
    creation_function:dict= {}

def edit_dictionary(
    d: dict, keys_to_remove: list = [], to_update: dict = {}
) -> dict:
    """
    Creates a copy of a dictionary while removing references to specific keys

    Parameters
    ----------
    d : dict
        Distionary to copy without a particular list
    keys : list
        List of keys to be excluded in the returned dictionary
    to_update : dict
        key to update from original dictionary
    Returns
    -------
    dict
        copy of the dictionary provided as an argument, excluding any references to the specified keys

    """
    copy_dict = d.copy()
    for key in keys_to_remove:
        if key in d:
            del copy_dict[key]
    copy_dict.update(to_update)
    return copy_dict


def same_json_or_remove(
    parameters: dict, output: str, verbose: bool
) -> bool:
    """
    If the file exist and the json is the same returns True.
    Else: removes the file and the json associated with it.

    Parameters
    ----------
    parameters : dict
        Parameter dictionary of executed functions.
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
                parameters, json_file, verbose
            ):
                if verbose:
                    print(f"File {output} already created with these parameters")
                return True
        os.remove(json_file)
    if os.path.exists(output):
        os.remove(output)
    return False


def check_same_existing_json(
    parameters: dict, json_file: str, verbose: bool
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
        if key not in prev_parameters:
            if verbose:
                print(f"new parameter {key}")
            return False
        elif prev_parameters[key] != value:
            if verbose:
                print(f"different {key}: old:{prev_parameters[key]}, new:{value}")
            return False

    # Check dates for all input files dates are consistent
    input_files = parameters['input_files']

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
    file_outputs: list,
    dir_name: str
    ) -> None:
    """
    Removes the original file path and the asociated json file

    Parameters
    ----------
    params : dict
        Parameter dictionary of executed functions. Paths should be relative.
    file_outputs :  list
    dirname : str
        Combine the dirname with the basename to obtain the absolute path.
    Returns
    -------
    None

    """

    data = params.copy()
    temp_date_str = ''
    for output in file_outputs:
        log_path = json_filename(output)
        actual_date = datetime.now(datetime.timezone.utc)
        data["input_modification_date"] = None
        for input_file in data['input_files']:
            input_json = json_filename(input_file)
            if os.path.exists(os.path.join(dir_name, input_json)):
                with open(os.path.join(dir_name, input_json)) as file:
                    input_data = json.load(file)
                temp_date_str = max(input_data["date"], temp_date_str)
        data["input_modification_date"] = temp_date_str
        data["date"] = actual_date.strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(dir_name, log_path), "w") as file:
            json.dump(data, file, indent=4)
