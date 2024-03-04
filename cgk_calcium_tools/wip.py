import os
from pathlib import Path
from glob import glob
import numpy as np
import isx
import json
from typing import Union, Tuple
from .processing import fix_frames
import pandas as pd
import shutil
from .files_io import (
    write_log_file,
    remove_file_and_json,
    same_json_or_remove,
    json_filename,
)
from typing import Iterable


def ifstr2list(x) -> list:
    if isinstance(x, list):
        return x
    return [x]


class isx_files_handler:
    def __init__(
        self,
        main_data_folder: Union[str, list] = ".",
        outputsfolders: Union[str, list] = ".",
        data_subfolders: Union[str, list] = ".",
        files_patterns: Union[str, list] = ".isx",
        processing_steps: list = ["PP", "TR", "BP", "MC"],
        one_file_per_folder: bool = True,
        recording_labels: Union[list, None] = None,
        files_list_log: Union[str, None] = None,  # capaz no se necesite mas....
        check_new_imputs: bool = True,
        parameters_path: str = os.path.join(
            os.path.dirname(__file__), "default_parameter.json"
        ),
        overwrite_metadata: bool = False,
    ) -> None:
        self.processing_steps = processing_steps
        assert os.path.exists(parameters_path), "parameters file does not exist"
        assert (
            len([s for s in processing_steps if s.startswith("TR")]) <= 1
        ), "Pipeline can't handle multiple trims"
        with open(parameters_path) as file:
            self.default_parameters = json.load(file)

        # if files_list_log is not None:
        #     input_parameters = {
        #         "main_data_folder": main_data_folder,
        #         "outputsfolders": outputsfolders,
        #         "data_subfolders": data_subfolders,
        #         "files_patterns": files_patterns,
        #         "one_file_per_folder": one_file_per_folder,
        #         "recording_labels": recording_labels,
        #     }

        #     if os.path.exists(files_list_log):
        #         with open(files_list_log, "r") as file:
        #             file_handler_status = json.load(file)

        #         if file_handler_status["input_parameters"] == input_parameters:
        #             self.__dict__.update(file_handler_status["computed_parameters"])
        #             return
        #         else:
        #             os.remove(files_list_log)
        recording_labels_iter = iter(recording_labels)
        lists_inputs = {
            "main_data_folder": ifstr2list(main_data_folder),
            "outputsfolders": ifstr2list(outputsfolders),
            "data_subfolders": ifstr2list(data_subfolders),
            "files_patterns": ifstr2list(files_patterns),
        }
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

        self.rec_subfolders = []
        self.outputsfolders = []
        self.rec_paths = []
        self.p_rec_paths = []
        self.p_outputsfolders = []
        self.p_recording_labels = []
        self.focus_files = {}
        self.efocus = []
        meta = {
            "rec_subfolders": [],
            "outputsfolders": [],
            "recording_labels": [],
            "rec_paths": [],
            "p_rec_paths": [],
            "p_outputsfolders": [],
            "p_recording_labels": [],
            "focus_files": {},
            "efocus": [],
        }
        for mainf, subfolder, fpatter, outf in zip(
            lists_inputs["main_data_folder"],
            lists_inputs["data_subfolders"],
            lists_inputs["files_patterns"],
            lists_inputs["outputsfolders"],
        ):
            if check_new_imputs:
                # get_from_nas
                files = glob(str(Path(mainf) / subfolder / fpatter), recursive=False)
                assert len(files) > 0, "No file found for {}.".format(
                    str(Path(mainf) / subfolder / fpatter)
                )
                metadata = {}

                if one_file_per_folder:
                    assert len(files) == 1, "Multiple files found for {}.".format(
                        str(Path(mainf) / subfolder / fpatter)
                    )
                    # self.rec_paths.append(
                    #     files[0]
                    # )
                    # aux_rec_path = files  # creoq ue no la necesito

                    # aux_outputsfolders = [str(Path(outf) / f)]
                    # self.outputsfolders.append(aux_outputsfolders)
                    # metadata[files[0]]['outputsfolders'] = aux_outputsfolders

                else:
                    files = [
                        r for r in files if r not in self.rec_paths
                    ]  # antes habia una f...
                    # self.rec_paths.extend(files) # creoq ue no
                    # aux_rec_path = files  # creoq ue no la necesito
                    # aux_outputsfolders = [
                    #     str(Path(outf) / subfolder) for i in range(len(files))
                    # ]
                    # self.outputsfolders.extend(aux_outputsfolders)
                for file in files:
                    metadata[file] = {
                        "rec_subfolders": [],
                        "outputsfolders": [str(Path(outf) / subfolder)],
                        "recording_labels": [],
                        "rec_paths": [file],
                        "p_rec_paths": [],
                        "p_outputsfolders": [],
                        "p_recording_labels": [],
                        "focus_files": {},
                        "efocus": [],
                    }

                    if recording_labels is None:
                        metadata[file]["recording_labels"] = metadata[file]["rec_paths"]
                    else:
                        assert not one_file_per_folder, "Multiple files found with {}. Recording labels not supported.".format(
                            str(Path(mainf) / subfolder / fpatter)
                        )
                        metadata[file]["recording_labels"] = next(recording_labels_iter)

                    # # Lookig for multiplanes:
                    # for rec in enumerate(
                    #     aux_rec_path
                    # ):  # puede ir files directo y borro aux_Rec_path
                    raw_gpio_file = (
                        os.path.splitext(file)[0] + ".gpio"
                    )  # raw data for gpio
                    updated_gpio_file = (
                        os.path.splitext(file)[0] + "_gpio.isxd"
                    )  # after the first reading gpio is converted to this
                    local_updated_gpio_file = os.path.join(
                        metadata[file]["outputsfolders"][0],
                        Path(
                            updated_gpio_file
                        ).name,  # aca estaba self.outputsfolders[i]
                    )  # new gpio copied in output
                    if os.path.exists(local_updated_gpio_file):
                        efocus = get_efocus(local_updated_gpio_file)
                    elif os.path.exists(updated_gpio_file):
                        efocus = get_efocus(updated_gpio_file)
                    elif os.path.exists(raw_gpio_file):
                        local_raw_gpio_file = os.path.join(
                            metadata[file]["outputsfolders"][0],
                            Path(
                                raw_gpio_file
                            ).name,  # aca estaba aux_putputsfolders[i]
                        )
                        shutil.copy2(raw_gpio_file, local_raw_gpio_file)
                        efocus = get_efocus(local_raw_gpio_file)
                    else:
                        video = isx.Movie.read(file)
                        get_acquisition_info = video.get_acquisition_info().copy()
                        del video  # usefull for windows
                        if "Microscope Focus" in get_acquisition_info:
                            assert isx.verify_deinterleave(
                                file, get_acquisition_info["Microscope Focus"]
                            ), f"Info {file}: Multiple Microscope Focus but not gpio file"
                            efocus = [get_acquisition_info["Microscope Focus"]]
                        else:
                            efocus = [0]
                            print(
                                f"Info: Unable to verify Microscope Focus config in: {file}"
                            )
                        if len(efocus) == 1:
                            metadata[file]["focus_files"][file] = [file]
                            metadata[file]["p_rec_paths"].append(file)
                            metadata[file]["p_recording_labels"].append(
                                metadata[file]["recording_labels"][0]
                            )
                            metadata[file]["efocus"].append(efocus)
                        else:
                            metadata[file]["efocus"].append(efocus)
                            pr = Path(file)
                            efocus_filenames = [
                                pr.stem + "_" + str(ef) + pr.suffix for ef in efocus
                            ]
                            self.focus_files[file] = [
                                str(
                                    Path(metadata[file]["outputsfolders"][0]) / Path(ef)
                                )
                                for ef in efocus_filenames
                            ]
                            self.p_recording_labels.extend(
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

                for path, intern_data in metadata.items():
                    json_file = os.path.splitext(path)[0] + "_metadata.json"
                    # if exist, is remove to write it again with the new data
                    if os.path.exists(json_file):
                        os.remove(json_file)
                    with open(json_file, "w") as j_file:
                        json.dump(intern_data, j_file)
                    for key, _ in intern_data.items():
                        meta[key].extend(metadata[file][key])
            else:
                fpatter = "_metadata.json"
                files = glob(str(Path(outf) / subfolder / fpatter), recursive=False)
                for f in files:
                    with open(f) as file_data:
                        for key, path in file_data:
                            meta[key].extend(file_data[path[key]])

            self.__dict__.update(meta)

        for ofolder in self.outputsfolders:
            os.makedirs(ofolder, exist_ok=True)
