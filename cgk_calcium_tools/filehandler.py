import copy
import os
from pathlib import Path
from glob import glob
import warnings
import numpy as np
import isx
import json
from typing import Union, Tuple, Iterable
from .processing import fix_frames
import pandas as pd
import shutil
from .files_io import (
    write_log_file,
    remove_file_and_json,
    same_json_or_remove,
    json_filename,
)
from .jupyter_outputs import progress_bar
from time import perf_counter
from datetime import timedelta


def ifstr2list(x) -> list:
    if isinstance(x, list):
        return x
    return [x]


def timer(method):
    def timed(self, *args, **kwargs):
        start_time = perf_counter()
        result = method(self, *args, **kwargs)
        end_time = perf_counter()
        elapsed_time = end_time - start_time
        print(f'{method.__name__} executed in {timedelta(seconds=elapsed_time)}')
        self.total_time += elapsed_time
        return result
    return timed


class isx_files_handler:
    """
    This class helps handle and process Inscopix files/movies. 

    Parameters
    ----------
    main_data_folder : str or list, optional
        Root folder containing the data. If a list each element should correspond
        with elements in 'data_subfolders' and 'files_pattern'. Output data follows
        the folder structure after this root folder. By default "."
    data_subfolders : str or list, optional
        Subfolders containing the files specified by 'files_patterns'. By default "."
    files_patterns : list, optional
        Naming patterns for the files. Also an easy way to select for one or
        multiple files from the same folder. By default "**/*.isx" (selects all)
    outputsfolders : str or list, optional
        Folder where the outputs are saved, following the file structure after 
        'main_data_folder'. By default "."
    processing_steps: list, optional
        List of processing steps to run, listed in order 
        Naming steps will be use, adding one affter the previous ones. By default "["PP", "TR", "BP", "MC"]"
    single_file_match: bool, optional
        If True the pipeline will expect one .isx file per folder listed in the 'files_patterns' 
        variable. By default "False"
    recording_labels: list, optional
        Name the recorded file with a label to make it easier to recognize. By default "None"
    files_list_log: str or None, optional
        Path where the previous processing could have happened. If it has been
        made previously, there is an early return. By default "None"
    parameters_path: str, optional
        Path with the information of the default parameter. If it does not exist, an error occurs.
        By default "default_parameter.json" in the same folder as this file.
    overwrite_metadata: bool, optional
        If True overwrite metadata json file. By default False.
    skip_pattern: str, optional
        String pattern to ignore certain files. Case-sensitive. Default is None
    """
    
    total_time = 0

    @timer
    def __init__(
        self,
        main_data_folder: Union[str, list] = ".",
        outputsfolders: Union[str, list] = ".",
        data_subfolders: Union[str, list] = ".",
        files_patterns: Union[str, list] = "**/*.isxd",
        processing_steps: list = ["DI", "PP", "TR", "BP", "MC", "DFF", "PM"],
        single_file_match: bool = False,
        recording_labels: Union[list, None] = None,
        check_new_inputs: bool = True,
        parameters_path: str = os.path.join(
            os.path.dirname(__file__), "default_parameter.json"
        ),
        overwrite_metadata: bool = False,
        skip_pattern: str = None,
    ):
        
        self.processing_steps = processing_steps
        self.deinterleave_output_files = []
        
        # Check if the parameters file exists at the given path, open and load it. 
        assert os.path.exists(parameters_path), "parameters file does not exist"
        with open(parameters_path) as file:
            self.default_parameters = json.load(file)
        
        # Check if step TR is listed more than once 
        assert (
            len([s for s in processing_steps if s.startswith("TR")]) <= 1
        ), "Pipeline can't handle multiple trims"

        # Initialize iterator for recording labels if provided 
        if recording_labels is not None:
            recording_labels_iter = iter(recording_labels)
        
        # Convert string inputs to lists 
        lists_inputs = {
            "main_data_folder": ifstr2list(main_data_folder),
            "outputsfolders": ifstr2list(outputsfolders),
            "data_subfolders": ifstr2list(data_subfolders),
            "files_patterns": ifstr2list(files_patterns),
        }

        # Ensure all list inputs are the same length
        len_list_variables = np.unique([len(v) for v in lists_inputs.values()])
        len_list_variables = len_list_variables[len_list_variables > 1]

        if len(len_list_variables) > 0:
            assert (
                len((len_list_variables)) == 1
            ), "the list inputs should have the same length"
            len_list = len_list_variables[0]
        else:
            len_list = 1

        for k, v in lists_inputs.items():
            if len(v) != len_list:
                # this will extend the single inputs
                lists_inputs[k] = v * len_list
        
        meta = {
            "main_data_folder": lists_inputs['main_data_folder'],
            "outputsfolders": [],
            "recording_labels": [],
            "rec_paths": [],
            "p_rec_paths": [],
            "p_outputsfolders": [],
            "p_recording_labels": [],
            "focus_files": {},
            "efocus": [],
            "resolution": [],
            "duration": [],
            "frames_per_second": [],
        }

        loaded_meta_files = [] # Used to avoid loading the same json file multiple times
        
        for mainf, subfolder, fpatter, outf in zip(
            lists_inputs["main_data_folder"],
            lists_inputs["data_subfolders"],
            lists_inputs["files_patterns"],
            lists_inputs["outputsfolders"],
        ):
            if check_new_inputs:
                # Grab all the files matching the input parameters 
                allFiles = glob(str(Path(mainf) / subfolder / fpatter),  recursive=True) #grabs all the files with fpatter.
                # Filter out files matching the skip pattern
                if skip_pattern is not None:
                    files = [file for file in allFiles if skip_pattern not in Path(file).name] #filter skip_pattern files out
                else:
                    files = allFiles
                
                # Error if no files are found, lists if files were skipped
                assert len(files) > 0, f"No file(s) found for {str(Path(mainf) / subfolder / fpatter)}, {len(allFiles)-len(files)} files skipped"
                
                # Prints confirmation of number of files found and skipped. 
                print(f'{len(files)} file(s) found, {len(allFiles)-len(files)} file(s) skipped')

                metadata = {}

                # If true, ensures only one file is found
                if single_file_match:
                    assert len(files) == 1, "Multiple files found for {}.".format(
                        str(Path(mainf) / subfolder / fpatter)
                    )
                else:
                    # removes files already in metadata
                    files = [r for r in files if r not in meta["rec_paths"]]
                
                # Initializes progress bar
                pb = progress_bar(len(files), 'Loading')
                                
                for file in files:
                    # skip processing if metadata file already exists and overwrite is not allowed
                    if not overwrite_metadata:
                        json_file = os.path.join(
                            str(Path(outf) / subfolder),
                            os.path.splitext(os.path.basename(file))[0]
                            + "_metadata.json",
                        )
                        if os.path.exists(json_file):
                            continue
                    
                    # Read the video file and grab metadata information
                    video = isx.Movie.read(file)
                    metadata[file] = copy.deepcopy(meta)
                    metadata[file]["outputsfolders"] = [str(Path(outf) / subfolder)]
                    metadata[file]["rec_paths"] = [file]
                    metadata[file]["resolution"] = [video.spacing.num_pixels]
                    metadata[file]["duration"] = [
                        video.timing.num_samples * video.timing.period.to_usecs() / 1e6
                    ]
                    metadata[file]["frames_per_second"] = [
                        1 / (video.timing.period.to_usecs() / 1e6)
                    ]

                    # Assign recording labels, if true -> same as rec_paths
                    # Else use input recording labels 
                    if recording_labels is None:
                        metadata[file]["recording_labels"] = metadata[file]["rec_paths"]
                    else:
                        assert (
                            single_file_match
                        ), "Multiple files found with {}. Recording labels not supported.".format(
                            str(Path(mainf) / subfolder / fpatter)
                        )
                        metadata[file]["recording_labels"] = next(recording_labels_iter)

                    # Lookig for multiplanes:
                    for ofolder in metadata[file]["outputsfolders"]:
                        os.makedirs(ofolder, exist_ok=True)
                    raw_gpio_file = (
                        os.path.splitext(file)[0] + ".gpio"
                    )  # raw data for gpio
                    updated_gpio_file = (
                        os.path.splitext(file)[0] + "_gpio.isxd"
                    )  # after the first reading gpio is converted to this
                    local_updated_gpio_file = os.path.join(
                        metadata[file]["outputsfolders"][0],
                        Path(updated_gpio_file).name,
                    )  # new gpio copied in output
                    if os.path.exists(local_updated_gpio_file):
                        efocus = get_efocus(local_updated_gpio_file)
                    elif os.path.exists(updated_gpio_file):
                        efocus = get_efocus(updated_gpio_file)
                    elif os.path.exists(raw_gpio_file):
                        local_raw_gpio_file = os.path.join(
                            metadata[file]["outputsfolders"][0],
                            Path(raw_gpio_file).name,
                        )
                        shutil.copy2(raw_gpio_file, local_raw_gpio_file)
                        efocus = get_efocus(local_raw_gpio_file)
                    else:
                        get_acquisition_info = video.get_acquisition_info().copy()
                        if "Microscope Focus" in get_acquisition_info:
                            if not isx.verify_deinterleave(
                                file, get_acquisition_info["Microscope Focus"]
                            ):
                                warnings.warn(
                                    f"Info {file}: Multiple Microscope Focus but not gpio file",
                                    Warning,
                                )
                                efocus = [0]
                            else:
                                efocus = [get_acquisition_info["Microscope Focus"]]
                        else:
                            efocus = [0]
                            print(
                                f"Info: Unable to verify Microscope Focus config in: {file}"
                            )
                    video.flush()
                    del video  # usefull for windows
                    
                    # Update metadata with focus data
                    if len(efocus) == 1:
                        metadata[file]["focus_files"][file] = [file]
                        metadata[file]["p_rec_paths"].append(file)
                        metadata[file]["p_recording_labels"].append(
                            metadata[file]["recording_labels"][0]
                        )
                        metadata[file]["efocus"].extend(efocus)
                    else:
                        metadata[file]["efocus"].extend(efocus)
                        pr = Path(file)
                        efocus_filenames = [
                            pr.stem + "_" + str(ef) + pr.suffix for ef in efocus
                        ]
                        metadata[file]["focus_files"][file] = [
                            str(Path(metadata[file]["outputsfolders"][0]) / Path(ef))
                            for ef in efocus_filenames
                        ]
                        metadata[file]["p_recording_labels"].extend(
                            [
                                metadata[file]["recording_labels"][0] + f"_{ef}"
                                for ef in efocus
                            ]
                        )
                        metadata[file]["p_rec_paths"].extend(
                            metadata[file]["focus_files"][file]
                        )
                    metadata[file]["p_outputsfolders"].extend(
                        [metadata[file]["outputsfolders"][0]] * len(efocus)
                    )

                    # Update progress bar
                    pb.update_progress_bar(1)
                    
                
                # Save metadata to JSON files
                for raw_path, intern_data in metadata.items():
                    json_file = os.path.join(
                        intern_data["outputsfolders"][0],
                        os.path.splitext(os.path.basename(raw_path))[0]
                        + "_metadata.json",
                    )
                    # if exist, is remove to write it again with the new data
                    if os.path.exists(json_file):
                        os.remove(json_file)
                    with open(json_file, "w") as j_file:
                        json.dump(intern_data, j_file)
            else:
                assert not overwrite_metadata, "Overwriting json file not possible"
            
            # Load metadata from existing JSON files
            base_folder = os.path.join(outf, subfolder)
            files = glob(
                os.path.join(
                    base_folder, os.path.splitext(fpatter)[0] + "_metadata.json"
                ),
                recursive = True
            )


            for f in files:
                if f in loaded_meta_files:
                    continue
                with open(f, "r") as file_data:
                    json_data = json.load(file_data)
                    for key_j, value_j in json_data.items():
                        if key_j == "focus_files":
                            meta[key_j].update(value_j)
                        elif key_j == "efocus":
                            meta[key_j].append(value_j)
                        else:
                            meta[key_j].extend(value_j)
                loaded_meta_files.append(f)
        # Update object dictionary with metadata
        self.__dict__.update(meta)

    def get_pair_filenames(self, operation: str) -> Tuple[list, list]:
        """
        This method returns the input/output pairs for the given operation step.

        Parameters
        ----------
        operation : str
            Operation in the steps.

        Returns
        -------
        Tuple[list,list]
            pair of input and output file paths
        """

        # Checks if operation is in the lsit of processing steps
        assert (
            operation in self.processing_steps
        ), f"operation must be in the list {self.processing_steps}, got: {operation}"

        # Copy the list of processing steps to a local variable
        local_processing_steps = self.processing_steps
        
        # Remove DI from list, if its first
        if local_processing_steps[0] == "DI":
            local_processing_steps.remove("DI")

        opi = local_processing_steps.index(operation)

        outputs = []
        inputs = []

        suffix_out = "-" + "-".join(local_processing_steps[: opi + 1]) + ".isxd"
        
        if opi == 0:
            suffix_in = None
            #inputs = self.p_rec_paths
            inputs = self.deinterleave_output_files
        else:
            suffix_in = "-" + "-".join(local_processing_steps[:opi]) + ".isxd"

        for ofolder, file in zip(self.p_outputsfolders, self.p_rec_paths):
            outputs.append(str(Path(ofolder, Path(file).stem + suffix_out)))
            if suffix_in is not None:
                inputs.append(str(Path(ofolder, Path(file).stem + suffix_in)))
        return len(inputs), zip(inputs, outputs)

    def get_filenames(self, op: Union[None, str] = None) -> list:
        """
        This method return the filenames of the files create after the operation "op".

        Parameters
        ----------
        op : str or None, optional
            Operation in the steps. If None, it will be the final one. Default: None

        Returns
        -------
        list:
            With the outputs file path

        """
        
        local_processing_steps = self.processing_steps
        
        if "DFF" in local_processing_steps:
            local_processing_steps.remove("DFF")
        if "PM" in local_processing_steps:
            local_processing_steps.remove("PM")
        
        if op is None:
            op = self.processing_steps[-1]
        assert (
            op in self.processing_steps
        ), f"operation must be in the list {self.processing_steps}, got: {op}"

        opi = self.processing_steps.index(op)
        outputs = []
        suffix_out = "-" + "-".join(self.processing_steps[: opi + 1]) + ".isxd"

        for ofolder, file in zip(self.p_outputsfolders, self.p_rec_paths):
            outputs.append(str(Path(ofolder, Path(file).stem + suffix_out)))
        return outputs

    def check_str_in_filenames(self, strlist: list, only_warning: bool = True) -> None:
        """
        This method verify the order of the files. Checking that each element of the
        input list is included in the recording filenames

        Parameters
        ----------
        strlist : list
            List of strings to check
        only_warning: bool, optional
            If False, throw an exception. Default, True.

         Returns
        -------
        None

        """
        for x, y in zip(strlist, self.rec_paths):
            if only_warning and x not in y:
                print(f"Warning: {x} not in {y}")
            else:
                assert x in y, f"Error {x} not in {y}"

    def get_results_filenames(
        self,
        name: str,
        op: Union[None, str] = None,
        subfolder: str = "",
        idx: Union[None, int] = None,
        single_plane: bool = True,
    ) -> list:
        """
        Returns a list with output files paths

        Parameters
        ----------
        name : str
            refers to the output path name.
        op : str or None, optional
            refers to the procesing steps to be executed. Default None.
        subfolder : str, optional
            string specifying the subfolder path. Default empty ("")
        idx : int or None, optional
            Default None.
        single_plane : bool, optional
            If true, it returns the name for each plane; otherwise, it returns one string per recording. Default "True".

        Returns
        -------
        list
            output paths

        """
        local_processing_steps = self.processing_steps

        if "DFF" in local_processing_steps:
            local_processing_steps.remove("DFF")
        if "PM" in local_processing_steps:
            local_processing_steps.remove("PM")
        if op is not None:
            opi = self.processing_steps.index(op)
            steps = self.processing_steps[: (opi + 1)]
        else:
            steps = self.processing_steps
        if "." in name:
            ext = ""
        else:
            ext = ".isxd"
        suffix_out = "-" + "-".join(steps) + "-{}{}".format(name, ext)

        if single_plane:
            file_list = self.p_rec_paths
            ofolders = self.p_outputsfolders
        else:
            file_list = self.rec_paths
            ofolders = self.outputsfolders
        outputs = []
        for i, (file, ofolder) in enumerate(zip(file_list, ofolders)):
            if idx is None or i in idx:
                os.makedirs(os.path.join(ofolder, subfolder), exist_ok=True)
                outputs.append(
                    str(
                        Path(
                            ofolder,
                            subfolder,
                            Path(file).stem + suffix_out,
                        )
                    )
                )
        return outputs

    @timer
    def remove_output_files(self, op, keep_json: bool = True) -> None:
        """
        This function remove output files

        Parameters
        ----------
        op : str
            Preprocessing operation to run
        keep_json : bool, optional
            If True, it does not remove the json file associated with the output file which will be removed. By default True.

        Returns
        -------
        None

        """
        if op == 'DI':
            paths = self.deinterleave_output_files
        elif op == "DFF":
            paths = self.get_results_filenames("dff", op=None)
        elif op == "PM":
            paths = self.get_results_filenames("maxdff", op=None)
        else:
            paths = self.get_filenames(op)
        
        for path in paths:
            if os.path.exists(path):
                os.remove(path)
            if not keep_json:
                json_file = json_filename(path)
                if os.path.exists(json_file):
                    os.remove(json_file)
        print(f"{len(paths)} Files removed!")

    @timer
    def run_step(
        self,
        op: str,
        overwrite: bool = False,
        verbose=False,
        pairlist: Union[Tuple[list, list], None] = None,
        **kws,
    ) -> None:
        """
        This function executes the specified preprocessing operation.

        Parameters
        ----------
        op : str
            Preprocessing operation to run
        overwrite : bool, optional
            Remove results and recompute them, by default False
        verbose : bool, optional
            Show additional messages, by default False

        Returns
        -------
        None

        """

        if pairlist is None:
            if op == "MC":
                translation_files = self.get_results_filenames(
                    "translations.csv", op=op
                )
                crop_rect_files = self.get_results_filenames("crop_rect.csv", op=op)
            elif op == "DFF":
                input = self.get_filenames(op=None)
                output = self.get_results_filenames("dff", op=None)
                amount_of_files = len(input)
                pairlist = zip(input, output)
            elif op == "PM":
                input = self.get_results_filenames("dff", op=None)
                output = self.get_results_filenames("maxdff", op=None)
                amount_of_files = len(input)
                pairlist = zip(input, output)
            if op in ["PP","TR", "BP", "MC"]:
                amount_of_files, pairlist = self.get_pair_filenames(op)
        elif op == "MC":
            translation_files = [
                os.path.splitext(p[1])[0] + "-translations.csv" for p in pairlist
            ]
            crop_rect_files = [
                os.path.splitext(p[1])[0] + "-crop_rect.csv" for p in pairlist
            ]

        if overwrite:
            for _, output in pairlist:
                remove_file_and_json(output)

        steps_list = {
            "DI": "de-interleave",
            "PP": "preprocessing",
            "BP": "spatial_filter",
            "MC": "motion_correct",
            "TR": "trim",
            "DFF": "dff",
            "PM": "project_movie"
        }

        operation = [x for k, x in steps_list.items() if op.startswith(k)]
        assert len(operation) == 1, f"Step {op} not starts with {steps_list}."
        operation = operation[0]

        if op != 'DI':
            parameters = self.default_parameters[operation].copy()
            for key, value in kws.items():
                assert key in parameters, f"The parameter: {key} does not exist"
                parameters[key] = value
        
        #TASK: No need for this step at this point - Can you confirm Fernando?
        '''
        if self.processing_steps.index(op) == 0:  # if first step
            # check if all exist
            for finput, _ in self.get_pair_filenames(op):
                assert os.path.exists(
                    finput
                ), f"""File f{input} not exist:
                    Run .de_interleave() to De-interleave multiplane movies"""
        '''

        if op.startswith('DI'):
            print("De-interleaving movies, please wait...")
            self.deinterleave_output_files = de_interleave(focus_files = self.focus_files, efocus = self.efocus)
        if op.startswith("PP"):
            print("Preprocessing movies, please wait...")
            preprocess_step(pairlist, parameters, amount_of_files, verbose)
        if op.startswith("BP"):
            print("Applying bandpass filter, please wait...")
            spatial_filter_step(pairlist, parameters, amount_of_files, verbose)
        if op.startswith("MC"):
            print("Applying motion correction, Please wait...")
            motion_correct_step(
                translation_files, crop_rect_files, parameters, pairlist, amount_of_files, verbose)
        if op.startswith("TR"):
            print("Trim movies, Please wait...")
            trim_movie(pairlist, parameters, amount_of_files, verbose)
        if op.startswith("DFF"):
            print("Normalizing via DF/F0, Please wait...")
            create_dff(pairlist, parameters, amount_of_files, overwrite, verbose)
        if op.startswith("PM"):
            print("Projecting Movie, Please wait...")
            project_movie(pairlist, parameters, amount_of_files, overwrite, verbose)
        print("done")

    @timer
    def extract_cells(
        self,
        alg: str,
        overwrite: bool = False,
        verbose: bool = False,
        cells_extr_params: Union[dict, None] = None,
        detection_params: Union[dict, None] = None,
        accept_reject_params: Union[dict, None] = None,
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
        multiplane_params : Union[dict,None], optional
            Parameters for multiplane registration, by default None

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
                    json_file = json_filename(fout)
                    if os.path.exists(json_file):
                        os.remove(json_file)

        pb = progress_bar(len(inputs_files), 'Extracting Cells')

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
                    ["comments"],
                    {"input_movie_files": input, "output_cell_set_files": output},
                )
            )

            if not os.path.exists(output):
                print(
                    f"Warning: Algorithm {alg}, failed to create file: {output}.\n"
                    + "Empty cellmap created with its place"
                )
                movie = isx.Movie.read(input)
                cell_set = isx.CellSet.write(output, movie.timing, movie.spacing)
                image_null = np.zeros(cell_set.spacing.num_pixels).astype(np.float32)
                trace_null = np.zeros(cell_set.timing.num_samples).astype(np.float32)
                cell_set.set_cell_data(0, image_null, trace_null, "")
                cell_set.flush()
                del cell_set
                del movie

            write_log_file(
                parameters,
                os.path.dirname(output),
                {"function": alg},
                input_files_keys=["input_movie_files"],
                output_file_key="output_cell_set_files",
            )
            pb.update_progress_bar(1)
        if verbose:
            print("Cell extraction, done")
        # detect events
        ed_parameters = self.default_parameters["event_detection"].copy()
        if detection_params is not None:
            for key, value in detection_params.items():
                assert key in ed_parameters, f"The parameter: {key} does not exist"
                ed_parameters[key] = value

        pb = progress_bar(len(cellsets), "Detecting Events")
        for input, output in zip(
            cellsets, self.get_results_filenames(f"{cellsetname}-ED", op=None)
        ):
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
                continue
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
                cell_set = isx.CellSet.read(input)
                evset = isx.EventSet.write(output, cell_set.timing, [""])
                evset.flush()
                del evset
                del cell_set
            write_log_file(
                ed_parameters,
                os.path.dirname(output),
                {"function": "event_detection"},
                input_files_keys=["input_cell_set_files"],
                output_file_key="output_event_set_files",
            )
            pb.update_progress_bar(1)
        if verbose:
            print("Event detection, done")

        # accept reject cells
        ar_parameters = self.default_parameters["accept_reject"].copy()
        if accept_reject_params is not None:
            for key, value in accept_reject_params.items():
                assert key in ar_parameters, f"The parameter: {key} does not exist"
                ar_parameters[key] = value

        pb = progress_bar(len(cellsets), "Accepting/Rejecting Cells")
        for input_cs, input_ev, config_json in zip(
            cellsets,
            self.get_results_filenames(f"{cellsetname}-ED", op=None),
            self.get_results_filenames(f"{cellsetname}-accept_reject", op=None),
        ):
            new_data = {
                "input_cell_set_files": os.path.basename(input_cs),
                "input_event_set_files": os.path.basename(input_ev),
            }
            ar_parameters.update(new_data)
            try:
                isx.auto_accept_reject(
                    **parameters_for_isx(
                        ar_parameters,
                        ["comments"],
                        {
                            "input_cell_set_files": input_cs,
                            "input_event_set_files": input_ev,
                        },
                    )
                )
            except Exception as e:
                if verbose:
                    print(e)
            write_log_file(
                ar_parameters,
                os.path.dirname(config_json),
                {
                    "function": "accept_reject",
                    "config_json": os.path.basename(config_json),
                },
                input_files_keys=["input_cell_set_files", "input_event_set_files"],
                output_file_key="config_json",
            )
            pb.update_progress_bar(1)
        if verbose:
            print("accept reject cells, done")

    @timer
    def run_multiplate_registration(
        self,
        overwrite: bool = False,
        verbose: bool = False,
        detection_params: Union[dict, None] = None,
        accept_reject_params: Union[dict, None] = None,
        multiplane_params: Union[dict, None] = None,
        cellsetname: Union[str, None] = None,
    ) -> None:
        """
        This function run a multiplane registration, detect events,
        and auto accept_reject

        Parameters
        ----------
        cellsetname : str
            Cellset name used, usually: 'cnmfe' or 'pca-ica'
        overwrite : bool, optional
            Force compute everything again, by default False
        verbose : bool, optional
            Show additional messages, by default False
        detection_params : Union[dict,None], optional
            Parameters for event detection, by default None
        accept_reject_params : Union[dict,None], optional
            Parameters for automatic accept_reject cell, by default None
        multiplane_params : Union[dict,None], optional
            Parameters for multiplane registration, by default None

        Returns
        -------
        None

        """
        if len([True for x in self.focus_files.values() if len(x) > 1]) == 0:
            return
        ed_parameters = self.default_parameters["event_detection"].copy()
        ar_parameters = self.default_parameters["accept_reject"].copy()
        mpr_parameters = self.default_parameters["multiplane_registration"].copy()

        if multiplane_params is not None:
            for key, value in multiplane_params.items():
                assert key in mpr_parameters, f"The parameter: {key} does not exist"
                mpr_parameters[key] = value

        self.output_file_paths = []
        pb = progress_bar(len(self.focus_files.keys()), 'Running Multiplane Registration to')
        for main_file, single_planes in self.focus_files.items():
            if len(single_planes) == 1:  # doesn't have multiplane
                continue
            input_cell_set_files = self.get_results_filenames(
                f"{cellsetname}",
                op=None,
                idx=[self.p_rec_paths.index(f) for f in single_planes],
            )
            idx = [self.rec_paths.index(main_file)]
            output_cell_set_file = self.get_results_filenames(
                f"{cellsetname}", op=None, idx=idx, single_plane=False
            )[0]
            ed_file = self.get_results_filenames(
                f"{cellsetname}-ED", op=None, idx=idx, single_plane=False
            )[0]
            ar_cell_set_file = self.get_results_filenames(
                f"{cellsetname}-accept_reject", op=None, idx=idx, single_plane=False
            )[0]

            input_cell_set_file_names = [os.path.basename(file) for file in input_cell_set_files]
            new_data = {
                "input_cell_set_files": input_cell_set_file_names,
                "output_cell_set_file": os.path.basename(output_cell_set_file),
                "auto_accept_reject": os.path.basename(ar_cell_set_file),
            }
            mpr_parameters.update(new_data)

            if not same_json_or_remove(
                mpr_parameters,
                output=output_cell_set_file,
                verbose=verbose,
                input_files_keys=["input_cell_set_files", "auto_accept_reject"],
            ):
                input = []
                for i in input_cell_set_files:
                    cs = isx.CellSet.read(i)
                    n_input = cs.num_cells
                    accepted = 0
                    
                    for n in range(n_input):
                            if cs.get_cell_status(n) == "accepted":
                                accepted += 1
                                
                    if accepted != 0:
                        input.append(i)
                    cs.flush()
                    del cs  # isx keeps the file open otherwise
                if len(input) == 0:
                    print(
                        f"Warning: File: {output_cell_set_file} not generated.\n"
                        + "Empty cellmap created in its place"
                    )
                    cell_set_plane = isx.CellSet.read(input_cell_set_files[0])
                    cell_set = isx.CellSet.write(
                        output_cell_set_file,
                        cell_set_plane.timing,
                        cell_set_plane.spacing,
                    )
                    image_null = np.zeros(cell_set.spacing.num_pixels, dtype=np.float32)
                    trace_null = np.zeros(cell_set.timing.num_samples, dtype=np.float32)
                    cell_set.set_cell_data(0, image_null, trace_null, "")
                    cell_set.flush()
                    del cell_set
                    del cell_set_plane  # isx keeps the file open otherwise
                elif len(input) == 1:
                    shutil.copyfile(input[0], output_cell_set_file)
                else:
                    isx.multiplane_registration(
                        **parameters_for_isx(
                            mpr_parameters,
                            ["comments", "auto_accept_reject"],
                            {
                                "input_cell_set_files": input,
                                "output_cell_set_file": output_cell_set_file
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

            # event detection in registered cellset
            new_data = {
                "input_cell_set_files": os.path.basename(output_cell_set_file),
                "output_event_set_files": os.path.basename(ed_file),
            }
            ed_parameters.update(new_data)
            if not same_json_or_remove(
                ed_parameters,
                output=ed_file,
                verbose=verbose,
                input_files_keys=["input_cell_set_files"],
            ):
                try:
                    isx.event_detection(
                        **parameters_for_isx(
                            ed_parameters,
                            ["comments"],
                            {
                                "input_cell_set_files": output_cell_set_file,
                                "output_event_set_files": ed_file,
                            },
                        )
                    )
                except Exception as e:
                    print(
                        f"Warning: Event_detection, failed to create file: {ed_file}.\n"
                        + "Empty file created with its place"
                    )
                    cell_set = isx.CellSet.read(output_cell_set_file)
                    evset = isx.EventSet.write(ed_file, cell_set.timing, [""])
                    evset.flush()
                    del evset
                    del cell_set

                write_log_file(
                    ed_parameters,
                    os.path.dirname(output_cell_set_file),
                    {"function": "event_detection"},
                    input_files_keys=["input_cell_set_files"],
                    output_file_key="output_event_set_files",
                )
            # auto accept reject
            new_data = {
                "input_cell_set_files": os.path.basename(output_cell_set_file),
                "input_event_set_files": os.path.basename(ed_file),
            }
            ar_parameters.update(new_data)
            try:
                isx.auto_accept_reject(
                    **parameters_for_isx(
                        ar_parameters,
                        ["comments"],
                        {
                            "input_cell_set_files": output_cell_set_file,
                            "input_event_set_files": ed_file,
                        },
                    )
                )
            except Exception as e:
                if verbose:
                    print(e)
            write_log_file(
                ar_parameters,
                os.path.dirname(output_cell_set_file),
                {"function": "accept_reject", "config_json": ar_cell_set_file},
                input_files_keys=["input_cell_set_files", "input_event_set_files"],
                output_file_key="config_json",
            )
            self.output_file_paths.append(output_cell_set_file)
            pb.update_progress_bar(increment=1)
        print("done")

    def deconvolve_cellset():
        #isx.deconvolve_cellset()
        pass
    
    def cell_metrics(self, cellsetname: str, verbose=False) -> pd.DataFrame:
        """
        This function use the isx.cell_metrics function, which compute cell metrics
        for a given cell set and events combination

        Parameters
        ----------
        cellsetname : str
            cell label to get filename
        verbose : bool, optional
            Show additional messages, by default False

        Returns
        -------
        pd.DataFrame
            a concatenates list with metrics


        """
        cell_set_files = self.get_results_filenames(
            f"{cellsetname}", op=None, single_plane=False
        )
        ed_files = self.get_results_filenames(
            f"{cellsetname}-ED", op=None, single_plane=False
        )
        metrics_files = self.get_results_filenames(
            f"{cellsetname}_metrics.csv", op=None, single_plane=False
        )

        for cellset, ed, metric in zip(cell_set_files, ed_files, metrics_files):
            inputs_args = {
                "input_cell_set_files": os.path.basename(cellset),
                "input_event_set_files": os.path.basename(ed),
                "output_metrics_files": os.path.basename(metric),
            }
            if not same_json_or_remove(
                inputs_args,
                output=metric,
                verbose=verbose,
                input_files_keys=["input_cell_set_files", "input_event_set_files"],
            ):
                try:
                    isx.cell_metrics(
                        **parameters_for_isx(
                            inputs_args,
                            to_update={
                                "input_cell_set_files": cellset,
                                "input_event_set_files": ed,
                                "output_metrics_files": metric,
                            },
                        )
                    )
                except Exception as e:
                    print(e)
                write_log_file(
                    inputs_args,
                    os.path.dirname(metric),
                    {"function": "cell_metrics"},
                    input_files_keys=["input_cell_set_files", "input_event_set_files"],
                    output_file_key="output_metrics_files",
                )

        df = []
        for metric_file, label, cell_set_file in zip(
            metrics_files, self.recording_labels, cell_set_files
        ):
            aux = pd.read_csv(metric_file)
            aux["Recording Label"] = label
            cell_set = isx.CellSet.read(cell_set_file)
            num_cells = cell_set.num_cells
            status = pd.DataFrame.from_dict(
                {
                    cell: [cell_set.get_cell_name(cell), cell_set.get_cell_status(cell)]
                    for cell in range(num_cells)
                },
                orient="index",
                columns=["cellName", "status"],
            )
            cell_set.flush()
            aux = aux.merge(status, on="cellName")

            df.append(aux)

        return pd.concat(df)

    def _recompute_from_log(self, json_file: str) -> None:
        if not os.path.exists(json_file):
            assert os.path.exists(
                os.path.splitext(json_file)
            ), "Error: json file not found"
        with open(json_file) as file:
            data = json.load(file)
        if "input_movie_file" in data:
            input_key = "input_movie_file"
            output_key = "output_movie_file"
        else:  # different functions have different arguments
            input_key = "input_movie_files"
            output_key = "output_movie_files"

        input = os.path.join(os.path.dirname(json_file), data[input_key])

        if not os.path.exists(input):
            self._recompute_from_log(
                os.path.join(
                    os.path.dirname(json_file), os.path.splitext(input)[0] + ".json"
                )
            )
        # TASK needs to be updated
        steps_list = {
            "preprocess": "PP",
            "spatial_filter": "BP",
            "motion_correct": "MC",
            "trimming": "TR",
        }
        if data["function"] in steps_list:
            function = steps_list[data["function"]]
        else:
            function = data["function"]

        pairlist = [
            [
                os.path.join(os.path.dirname(json_file), data[input_key]),
                os.path.join(os.path.dirname(json_file), data[output_key]),
            ],
        ]
        del data[input_key]
        del data[output_key]

        del data["function"]
        del data["isx_version"]
        del data["input_modification_date"]
        del data["date"]
        self.run_step(
            op=function,
            overwrite=False,
            verbose=False,
            pairlist=pairlist,
            **data,
        )

    def recompute_from_log(self, op: str) -> None:
        """Recompute operation from logs (json files) inside the folders.
        If the output file already exists, it will be skipped.

        Parameters
        ----------
        op : str

            operation

        """

        outputs = self.get_filenames(op=op)
        for output in outputs:
            if not os.path.exists(output):
                json_file = os.path.splitext(output)[0] + ".json"

                self._recompute_from_log(json_file)
        print("done")

    def get_total_time(cls):
        print(f'Current total execution time {timedelta(seconds=cls.total_time)}')


def de_interleave(focus_files: dict, efocus: list, overwrite: bool = False) -> None:
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
    pb = progress_bar(len(focus_files), 'Deinterleaving')
    output_files = []
    for (main_file, planes_fs), focus in zip(focus_files.items(), efocus):
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

        # Update progress bar
        pb.update_progress_bar(1)
        output_files += planes_fs
    # Need to return value to save the output values back in the object for the remove function later used.
    return output_files


def preprocess_step(
    pairlist: Tuple[list, list], parameters: dict, amount_of_files: int, verbose: bool = False 
) -> None:
    """
    After performing checks, use the isx.preprocess function, which preprocesses movies,
    optionally applying spatial and temporal downsampling and cropping.

    Parameters
    ----------
    pairlist : Tuple[list, list]
        Tuple containing lists of input and output paths
    parameters : dict
       Parameter dictionary of executed functions.
    verbose : bool, optional
        Show additional messages, by default False

    Returns
    -------
    None

    """
    
    # Initialize progress bar
    pb = progress_bar(amount_of_files, 'Preprocessing')
    
    for input, output in pairlist:
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
            continue

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
        # Update progress bar
        pb.update_progress_bar(1)


def spatial_filter_step(
    pairlist: Tuple[list, list], parameters: dict, amount_of_files: int, verbose: bool = False
) -> None:
    """
    After performing checks, use the isx.spatial_filter function, which
    apply spatial bandpass filtering to each frame of one or more movies

    Parameters
    ----------
    pairlist: Tuple[list, list]
        Tuple containing lists of input and output paths
    parameters : dict
        Parameter dictionary of executed functions.
    verbose : bool, optional
        Show additional messages, by default False

    Returns
    -------
    None

    """
    
    # Initialize progress bar
    pb = progress_bar(amount_of_files, 'Applying Bandpass Filter to')

    for input, output in pairlist:
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
            continue
        isx.spatial_filter(
            **parameters_for_isx(
                parameters,
                ["comments"],
                {"input_movie_files": input, "output_movie_files": output},
            )
        )
        write_log_file(
            parameters,
            os.path.dirname(output),
            {"function": "spatial_filter"},
            input_files_keys=["input_movie_files"],
            output_file_key="output_movie_files",
        )
        if verbose:
            print("{} bandpass filtering completed".format(output))
        # Update progress bar
        pb.update_progress_bar(1)


def motion_correct_step(
    translation_files: list,
    crop_rect_files: list,
    parameters: dict,
    pairlist: Tuple[list, list],
    amount_of_files: int,
    verbose=False
) -> None:
    """
    After checks, use the isx.motion_correct function, which motion correct movies to a reference frame.

    Parameters
    ----------
    translation_files : list
        A list of file names to write the X and Y translations to. Must be either None,
        in which case no files are written, or a list of valid file names equal in
        length to the number of input and output file names
    crop_rect_files : list
        The path to a file that will contain the crop rectangle applied to the input
        movies to generate the output movies
    parameters : dict
       Parameter list of executed functions.
    pairlist : Tuple[list, list]
        Tuple containing lists of input and output paths
    verbose : bool, optional
        Show additional messages, by default False

    Returns
    -------
    None

    Examples
    --------
    """
    # Initialize progress bar
    pb = progress_bar(amount_of_files, 'Applying Motion Correction to')
    for i, (input, output) in enumerate(pairlist):
        new_data = {
            "input_movie_files": os.path.basename(input),
            "output_movie_files": os.path.basename(output),
            "output_translation_files": os.path.basename(translation_files[i]),
            "output_crop_rect_file": os.path.basename(crop_rect_files[i]),
        }
        parameters.update(new_data)
        if same_json_or_remove(
            parameters,
            input_files_keys=["input_movie_files"],
            output=output,
            verbose=verbose,
        ):
            continue
        isx.motion_correct(
            **parameters_for_isx(
                parameters,
                ["comments"],
                {
                    "input_movie_files": input,
                    "output_movie_files": output,
                    "output_translation_files": translation_files[i],
                    "output_crop_rect_file": crop_rect_files[i],
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
        # Update progress bar
        pb.update_progress_bar(1)


def trim_movie(
    pairlist: Tuple[list, list], user_parameters: dict, amount_of_files: int,  verbose: bool = False
) -> None:
    """
    After verifying that the user_parameters are correct and obtaining the maximum file frame,
    it invokes the isx.trim_movie function, which trims frames from a movie to generate a new movie

    Parameters
    ----------
    pairlist: Tuple[list, list]
        Tuple containing lists of input and output paths
    user_parameters : dict
        Parameter of movie len.
    verbose : bool, optional
        Show additional messages, by default False

    Returns
    -------
    None

    """
    
    assert (
        user_parameters["video_len"] is not None
    ), "Trim movie requires parameter video len"
    
    # Initialize progress bar
    pb = progress_bar(amount_of_files, 'Trimming')
    
    for input, output in pairlist:
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
            continue
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
        # Update progress bar
        pb.update_progress_bar(1)


def create_dff(
    pairlist: Tuple[list, list], 
    parameters: dict,
    amount_of_files: int, 
    overwrite=False, 
    verbose=False, 
    **kws) -> None:
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
    # Initialize progress bar
    pb = progress_bar(amount_of_files, 'Normalizing via DF/F0')

    if overwrite:
        for input, output in zip(pairlist):
            remove_file_and_json(output)

    for key, value in kws.items():
        assert key in parameters, f"The parameter: {key} does not exist"
        parameters[key] = value

    for input, output in pairlist:
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
            continue
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
        # Update progress bar
        pb.update_progress_bar(1)
    print("done")


def project_movie(
    pairlist: Tuple[list, list], 
    parameters: dict, 
    amount_of_files: int,
    overwrite=False,
    verbose=False,
    **kws,
) -> None:
    """
    This function applies isx.project_movie to project movies to a single statistic image.

    Parameters
    ----------
    input_name : str, optional
        Input file path, by default "dff"
    output_name : str, optional
        Output file path, by default "maxdff"
    operation : str, optional
        Preprocessing operation to check, by default None.
    overwrite : bool, optional
        Remove results and recompute them, by default False
    verbose : bool, optional
        Show additional messages, by default False

    Returns
    -------
    None

    """

    # Initialize progress bar
    pb = progress_bar(amount_of_files, 'Projecting Movies')

    if overwrite:
        #get rid of self
        for input, output in pairlist:
            remove_file_and_json(output)

    for key, value in kws.items():
        assert key in parameters, f"The parameter: {key} does not exist"
        parameters[key] = value

    #turn into pairlist
    for input, output in pairlist:
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
            continue
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
        # Update progress bar
        pb.update_progress_bar(1)
    print("done")


def get_segment_from_movie(
    inputfile: str,
    outputfile: str,
    borders: Iterable,
    keep_start_time=False,
    unit="minutes",
) -> None:
    """
    This function gets a segment of a video to create a new one. It's just an easy
    handler of isx.trim_movie, which trim frames from a movie to produce a new movie

    Parameters
    ----------
    inputfile : str
        Path of the input file to use
    outputfile : str
        Path of the output file to use
    borders: iterable
        With two elements of the borders in minutes (integers) of the segment.
        Where negative times means from the end like indexing in a numpy array
    keep_start_time : Bool, optional
        If true, keep the start time of the movie, even if some of its
        initial frames are to be trimmed, by default False
    unit: str
        the unit to use for border. It could be 'minutes' or 'seconds'

    Returns
    -------
    None

    Examples
    --------
    >>> get_segment_from_movie('inputfile.isxd','output.isxd',[-30, -1]) #last 30 minutes

    """
    assert len(borders) == 2, "borders must have two elements"
    assert unit in ["minutes", "seconds"], """unit could be 'minutes' or 'seconds'. """
    movie = isx.Movie.read(inputfile)
    if unit == "minutes":
        unitfactor = 60
    else:
        unitfactor = 1
    numframes = movie.timing.num_samples
    duration_unit = numframes * movie.timing.period.to_msecs() / (1000 * unitfactor)
    timing_period_unit = movie.timing.period.to_msecs() / (1000 * unitfactor)

    movie.flush()

    crop_segments = []
    assert (
        borders[0] > -duration_unit
    ), "asking for a segment from {minutes[0]} {unit} before the end, but video is {duration_minutes} {unit} long."
    assert (
        borders[0] < duration_unit
    ), "asking for a segment from {minutes[0]} {unit}, but video is {duration_minutes} {unit} long."
    assert (
        borders[1] < duration_unit
    ), "asking for a segment up to {minutes[0]} {unit}, but video is {duration_minutes} {unit} long."
    assert (
        borders[1] > -duration_unit
    ), "asking for a segment up to {minutes[0]} {unit} before the end, but video is {duration_minutes} {unit} long."

    # remove fist frames:
    # don't cut if the segments are from the beggining or just exactlt
    # duration before the end
    if borders[0] != 0 and borders[0] != -duration_unit:
        if borders[0] < 0:
            end1 = (duration_unit + borders[0]) / timing_period_unit - 1
        else:
            end1 = borders[0] / timing_period_unit - 1
        crop_segments.append([0, int(end1)])
    # remove last frames:
    # and don't cut if the segments are up to the end or just the exact
    # duration
    if borders[1] != -1 and borders[1] != duration_unit:
        if borders[1] < 0:
            start1 = (duration_unit + borders[1] + 1) / timing_period_unit + 1
        else:
            start1 = borders[1] / timing_period_unit + 1
        crop_segments.append([int(start1), numframes])

    if os.path.exists(outputfile):
        os.remove(outputfile)
    if len(crop_segments) == 0:
        print("no trim need it")
        return
    isx.trim_movie(
        input_movie_file=inputfile,
        output_movie_file=outputfile,
        crop_segments=np.array(crop_segments),
        keep_start_time=keep_start_time,
    )


def get_efocus(gpio_file: str) -> list:
    """
    Read the gpio set from a file and get the data associated.

    Parameters
    ----------
    gpio_file : str
        path

    Returns
    -------
    list
        list with the video_efocus

    """
    gpio_set = isx.GpioSet.read(gpio_file)
    efocus_values = gpio_set.get_channel_data(gpio_set.channel_dict["e-focus"])[1]
    efocus_values, efocus_counts = np.unique(efocus_values, return_counts=True)
    min_frames_per_efocus = 100
    video_efocus = efocus_values[efocus_counts >= min_frames_per_efocus]
    assert (
        video_efocus.shape[0] < 4
    ), f"{gpio_file}: Too many efocus detected, early frames issue."
    return [int(v) for v in video_efocus]


def parameters_for_isx(
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
        data_folder = project_file[:-5] + "_data"
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
                "{eventdet_path}": ev_det.replace("\\","/"),
                "{eventdet_name}": Path(ev_det).name,
                "{cellset_path}": cellset.replace("\\","/"),
                "{cellset_name}":  Path(cellset).name,
                "{DFF_path}": dff.replace("\\","/"),
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

