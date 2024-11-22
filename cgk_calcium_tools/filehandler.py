import copy
import os
from pathlib import Path
import shutil

from glob import glob
import numpy as np
import isx
import json
from typing import Union, Tuple
import pandas as pd
from .files_io import (
    write_log_file,
    remove_file_and_json,
    same_json_or_remove,
    json_filename,
    parameters_for_isx,
)
from .jupyter_outputs import progress_bar
from time import perf_counter
from datetime import timedelta
from .isx_aux_functions import (
    cellset_is_empty,
    create_empty_cellset,
    create_empty_events,
    get_efocus,
    ifstr2list,
)
from .pipeline_functions import f_register, f_message, de_interleave
from .analysis_utils import apply_quality_criteria, compute_metrics,get_events


def timer(method):
    def timed(self, *args, **kwargs):
        start_time = perf_counter()
        result = method(self, *args, **kwargs)
        end_time = perf_counter()
        elapsed_time = end_time - start_time
        print(f"{method.__name__} executed in {timedelta(seconds=elapsed_time)}")
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
        processing_steps: list = ["PP", "TR", "BP", "MC", "DFF", "PM"],
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
            "main_data_folder": lists_inputs["main_data_folder"],
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

        loaded_meta_files = (
            []
        )  # Used to avoid loading the same json file multiple times

        for mainf, subfolder, fpatter, outf in zip(
            lists_inputs["main_data_folder"],
            lists_inputs["data_subfolders"],
            lists_inputs["files_patterns"],
            lists_inputs["outputsfolders"],
        ):
            if check_new_inputs:
                # Grab all the files matching the input parameters
                allFiles = glob(
                    str(Path(mainf) / subfolder / fpatter), recursive=True
                )  # grabs all the files with fpatter.
                # Filter out files matching the skip pattern
                if skip_pattern is not None:
                    files = [
                        file for file in allFiles if skip_pattern not in Path(file).name
                    ]  # filter skip_pattern files out
                else:
                    files = allFiles

                # Error if no files are found, lists if files were skipped
                assert (
                    len(files) > 0
                ), f"No file(s) found for {str(Path(mainf) / subfolder / fpatter)}, {len(allFiles)-len(files)} files skipped"

                # Prints confirmation of number of files found and skipped.
                print(
                    f"{len(files)} file(s) found, {len(allFiles)-len(files)} file(s) skipped"
                )

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
                pb = progress_bar(len(files), "Loading")

                for file in files:
                    # skip processing if metadata file already exists and overwrite is not allowed
                    if not overwrite_metadata:
                        json_file = os.path.join(
                            str(Path(outf) / subfolder),
                            os.path.splitext(os.path.basename(file))[0]
                            + "_metadata.json",
                        )
                        if os.path.exists(json_file):
                            pb.update_progress_bar(1)
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

                    for ofolder in metadata[file]["outputsfolders"]:
                        os.makedirs(ofolder, exist_ok=True)

                    efocus = get_efocus(
                        file, metadata[file]["outputsfolders"][0], video
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
                recursive=True,
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
        opi = self.processing_steps.index(operation)

        outputs = []
        inputs = []

        suffix_out = "-" + "-".join(self.processing_steps[: opi + 1]) + ".isxd"

        if opi == 0:
            suffix_in = None
            inputs = self.p_rec_paths
        else:
            suffix_in = "-" + "-".join(self.processing_steps[:opi]) + ".isxd"

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

    def apply_quality_criteria(
        self,
        cellsetname: str,
        max_corr=0.9,
        min_skew=0.05,
        only_isx_accepted=True,
        overwrite=False,
        verbose=False
    ) -> pd.DataFrame:
        cell_set_files = self.get_results_filenames(
            f"{cellsetname}", op=None, single_plane=False
        )
        metrics_files = self.get_results_filenames(
            f"{cellsetname}-ED_metrics.csv", op=None, single_plane=False
        )
        status_files = self.get_results_filenames(
            f"{cellsetname}-ED_status.csv", op=None, single_plane=False
        )
        return apply_quality_criteria(
            cell_set_files, metrics_files, status_files, 
            max_corr=max_corr,
            min_skew=min_skew,
            only_isx_accepted=only_isx_accepted,
            overwrite=overwrite,
            verbose=verbose
        )


    def get_status(self, cellsetname: str):
        data = []
        status_files = self.get_results_filenames(f"{cellsetname}-ED_status.csv", op=None, single_plane=False)
        for events,statusf in zip(self.events,status_files):
            df = pd.read_csv(statusf, index_col=0)
            df['File'] = events
            data.append(df)
        return pd.concat(data)
        
    def get_events(self, cellsetname: str, cells_used="accepted"):

        event_det_files = self.get_results_filenames(
            f"{cellsetname}-ED", op=None, single_plane=False
        )
        cellset_files = self.get_results_filenames(
            f"{cellsetname}", op=None, single_plane=False
        )
        return get_events(cellset_files, event_det_files, cells_used="accepted")



    def compute_metrics(self, cellsetname: str, verbose=False) -> pd.DataFrame:
        """
        This function compute the  correlation matrix for the cell traces.

        Parameters
        ----------
        cellsetname : str
            cell label to get filename
        verbose : bool, optional
            Show additional messages, by default False

        Returns
        -------
        pd.DataFrame
            DataFrame with the correlation matrix
        """

        cell_set_files = self.get_results_filenames(
            f"{cellsetname}", op=None, single_plane=False
        )

        ed_files = self.get_results_filenames(
            f"{cellsetname}-ED", op=None, single_plane=False
        )
        metrics_files = self.get_results_filenames(
            f"{cellsetname}-ED_metrics.csv", op=None, single_plane=False
        )  # it depends on event detection

        # TODO: it could be merge with recording_labels
        return compute_metrics(cell_set_files, ed_files, metrics_files, verbose=verbose)

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
        if op == "DFF":
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
            if op == "DFF":
                input = self.get_filenames(op=None)
                output = self.get_results_filenames("dff", op=None)
                amount_of_files = len(input)
                pairlist = zip(input, output)
            elif op == "PM":
                input = self.get_results_filenames("dff", op=None)
                output = self.get_results_filenames("maxdff", op=None)
                amount_of_files = len(input)
                pairlist = zip(input, output)
            if op in ["PP", "TR", "BP", "MC"]:
                amount_of_files, pairlist = self.get_pair_filenames(op)

        if overwrite:
            for _, output in pairlist:
                remove_file_and_json(output)

        steps_list = {
            "PP": "preprocessing",
            "BP": "spatial_filter",
            "MC": "motion_correct",
            "TR": "trim",
            "DFF": "dff",
            "PM": "project_movie",
        }

        operation = [x for k, x in steps_list.items() if op.startswith(k)]
        assert len(operation) == 1, f"Step {op} not starts with {steps_list}."
        operation = operation[0]

        parameters = self.default_parameters[operation].copy()
        for key, value in kws.items():
            assert key in parameters, f"The parameter: {key} does not exist"
            parameters[key] = value

        if self.processing_steps.index(op) == 0:  # deinterleave needed
            pb = progress_bar(len(self.focus_files), "Deinterleaving")
            for (main_file, planes_fs), focus in zip(
                self.focus_files.items(), self.efocus
            ):
                de_interleave(main_file, planes_fs, focus)
                pb.update_progress_bar(1)
        if (
            op.startswith("PP")
            or op.startswith("BP")
            or op.startswith("MC")
            or op.startswith("DFF")
            or op.startswith("PM")
            or op.startswith("TR")
        ):
            pb = progress_bar(amount_of_files, f_message[operation])
            for input, output in pairlist:
                f_register[operation](input, output, parameters, verbose)
                pb.update_progress_bar(1)
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

        pb = progress_bar(len(inputs_files), "Extracting Cells")

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
        if detection_params is not None:
            for key, value in detection_params.items():
                assert key in ed_parameters, f"The parameter: {key} does not exist"
                ed_parameters[key] = value
        if accept_reject_params is not None:
            for key, value in accept_reject_params.items():
                assert key in ar_parameters, f"The parameter: {key} does not exist"
                ar_parameters[key] = value
        if multiplane_params is not None:
            for key, value in multiplane_params.items():
                assert key in mpr_parameters, f"The parameter: {key} does not exist"
                mpr_parameters[key] = value

        self.output_file_paths = []
        pb = progress_bar(
            len(self.focus_files.keys()), "Running Multiplane Registration to"
        )

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

            input_cell_set_file_names = [
                os.path.basename(file) for file in input_cell_set_files
            ]
            new_data = {
                "input_cell_set_files": input_cell_set_file_names,
                "output_cell_set_file": os.path.basename(output_cell_set_file),
                "auto_accept_reject": os.path.basename(ar_cell_set_file),
            }
            mpr_parameters.update(new_data)

            if overwrite:
                if os.path.exists(output_cell_set_file):
                    os.remove(output_cell_set_file)
                    json_file = json_filename(output_cell_set_file)
                    if os.path.exists(json_file):
                        os.remove(json_file)

            if not same_json_or_remove(
                mpr_parameters,
                output=output_cell_set_file,
                verbose=verbose,
                input_files_keys=["input_cell_set_files", "auto_accept_reject"],
            ):
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
                    create_empty_events(output_cell_set_file, ed_file)

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

    def run_deconvolution(
        self,
        overwrite: bool = False,
        verbose: bool = False,
        params: Union[dict, None] = None,
        cellsetname: Union[str, None] = None,
    ) -> None:
        """
        This function runs the deconvolution step, save the new cellsets with the updated traces,
        and the new event detecction

        Parameters
        ----------
        cellsetname : str
            Cellset name used, usually: 'cnmfe' or 'pca-ica'
        overwrite : bool, optional
            Force compute everything again, by default False
        verbose : bool, optional
            Show additional messages, by default False
        params : Union[dict,None], optional
            Parameters for deconvolution, by default None

        Returns
        -------
        None

        """
        parameters = self.default_parameters["deconvolution"].copy()

        if params is not None:
            for key, value in params.items():
                assert key in parameters, f"The parameter: {key} does not exist"
                parameters[key] = value

        self.output_file_paths = []
        cell_sets = self.get_results_filenames(
            f"{cellsetname}", op=None, single_plane=False
        )
        denoise_files = self.get_results_filenames(
            f"{cellsetname}-DNI", op=None, single_plane=False
        )
        ed_files = self.get_results_filenames(
            f"{cellsetname}-SPI", op=None, single_plane=False
        )
        pb = progress_bar(len(cell_sets), "Running Deconvolution Registration to")

        for cellset, denoise_file, ed_file in zip(cell_sets, denoise_files, ed_files):

            if overwrite:
                remove_file_and_json(denoise_file)
                remove_file_and_json(ed_file)
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

            pb.update_progress_bar(increment=1)
        print("done")

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
        print(f"Current total execution time {timedelta(seconds=cls.total_time)}")


