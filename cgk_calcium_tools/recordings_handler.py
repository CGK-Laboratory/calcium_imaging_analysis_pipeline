import os
import json
from typing_extensions import Self
import pandas as pd

from .files_io import (
    json_filename,
    RecordingFile,
    load_recording_file,
)
from .handlers_functions import flatt_lists,hierarchy_from_paths,update_nested_lists
from .jupyter_outputs import progress_bar
from datetime import datetime, timezone
from .pipeline_functions import rec_functions
from .params_tools import _global_parameters


class RecordingHandler:
    """
    This class helps handle and process Inscopix files/movies.

    Parameters
    ----------
    main_data_folder : str, optional
        Root folder containing the data. Output data follows
        the folder structure after this root folder. By default "."
    output_folder : str, optional
        Folder where the outputs are saved. By default "."
    """

    def __init__(self, main_data_folder, recordings, output_folder, label, infere_hierarchy=False):
        assert isinstance(recordings, list), "Recordings must be a list"
        self.output_folder = output_folder
        self.main_data_folder = main_data_folder
        self.creation_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        self.label = label
        if infere_hierarchy:
            recordings = hierarchy_from_paths(recordings)
        self.recordings = recordings
        self.recordings_list = flatt_lists(recordings)

    def save(self):
        with open(os.path.join(self.output_folder,f"{self.label}.json"), "w") as f:
            json.dump(self.__dict__, f)


    @classmethod
    def load(cls, label: str, output_folder: str = "."):
        with open(os.path.join(output_folder,f"{label}.json"), "r") as f:
            data = json.load(f)

        instance = cls.__new__(cls)
        instance.name = data["name"]

        instance.output_folder = output_folder #it could be moved
        instance.main_data_folder = data["main_data_folder"]
        instance.creation_date =  data["creation_date"]
        instance.label =  data["label"]
        instance.recordings = convert_json_to_recordings(data["recordings"])
        instance.recordings_list = data["recordings_list"]
        return instance

    def rm(self, keep_json):
        counter = 0
        for rec in self.recordings:
            file = os.path.join(rec.main_folder, rec.recording_file)
            additional_files = [os.path.join(rec.main_folder, af) for af in rec.additional_files]                
            for f in [file] + additional_files:
                if os.path.exists(f):
                    os.remove(f)
                    counter += 1
                if not keep_json:
                    json_file = json_filename(f)
                    if os.path.exists(json_file):
                        os.remove(json_file)
                        counter += 1
        print(f"{counter} files removed.")

    def get_recording_info(self):
        return pd.DataFrame(self.recordings_list)

    def get_source(self, function, parameters, label=None):
        pass

    def get_metadata(self, function, parameters, label=None):
        pass

    def apply(self, fun, suffix=None,verbose=False,
        **kws) -> Self:
        if suffix is None:
            label = rec_functions[fun]['suffix']
        parameters = _global_parameters["functions"][fun].copy()
        for key, value in kws.items():
            assert key in parameters, f"The parameter: {key} does not exist"
            parameters[key] = value

        # Run the selected operation
        pb = progress_bar(len(self.recordings_list), rec_functions[fun]['message'])
        changes = []
        for rec_input in self.recordings_list:
            new_filename, additional_outputs = rec_functions[fun]['function'](rec_input, suffix=suffix,output_folder=self.output_folder, 
                                                                         parameters=parameters, verbose=verbose)
            new_rec = load_recording_file(file=self.output_folder,
                                          main_file=new_filename,
                                          creation_function=fun,
                                          additional_files=additional_outputs,
                                          main_folder=rec_input.main_folder,
                                          source_files=[rec_input.file])
            changes[rec_input] = new_rec
            pb.update_progress_bar(1)


        new_hierarchy = update_nested_lists(self.recordings.copy(), changes)
        return RecordingHandler(self.main_data_folder, new_hierarchy, self.output_folder, label, infere_hierarchy=False)
    def _recompute_from_log(self, json_file: str) -> None:
        if not os.path.exists(json_file):
            assert os.path.exists(os.path.splitext(json_file)), (
                "Error: json file not found"
            )
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



def convert_json_to_recordings(recordings_list):
    new_list = []
    for item in recordings_list:
        if isinstance(item, list):
            new_list.append(convert_json_to_recordings(item))
        if isinstance(item, dict):
            new_list.append(RecordingFile(**item))
        else:
            new_list.append(item)
    return new_list