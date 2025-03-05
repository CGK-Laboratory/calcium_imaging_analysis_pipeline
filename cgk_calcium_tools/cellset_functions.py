import isx
from typing import Union,  Tuple
from .processing import fix_frames
from .isx_aux_functions import create_similar_empty_events, cellset_is_empty, create_similar_empty_cellset
from typing import Callable
from .files_io import (
    write_log_file,
    same_json_or_remove,
    append_to_filaname
)
import os
import numpy as np
import shutil

csf_register: dict[str, Callable] = {}
csf_message: dict[str, str] = {}
csf_label: dict[str, str] = {}


def register(name: str, message=None) -> None:
    
    if message is None:
        message = f"Running function: {name}"

    def decorator(func: Callable) -> None:
        assert isinstance(func, Callable)
        csf_register[name] = func
        csf_message[name] = message
    return decorator

@register('isx:event_detection',"Detecting Events")
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
        create_similar_empty_events(input, output)
    write_log_file(
        ed_parameters,
        os.path.dirname(output),
        {"function": "event_detection"},
        input_files_keys=["input_cell_set_files"],
        output_file_key="output_event_set_files",
    )


@register(['isx:deconvolve_cellset'],"Running Deconvolution Registration")
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
            create_similar_empty_events(cellset, ed_file)
            create_similar_empty_cellset(cellset, denoise_file)
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

@register(['isx:multiplane_registration'],"Multiplane Registration...")
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
        create_similar_empty_cellset(
            input_file=input_cell_set_files[0],
            output_cell_set_file=output_cell_set_file,
        )

    elif len(input_cellsets) == 1:
        shutil.copyfile(input_cellsets[0], output_cell_set_file)
    else:
        isx.multiplane_registration(
            **parameters_for_isx(
                mpr_parameters,
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



@register('isx:cnmfe',"Detecting Events...",'ED')
def dectect_events():
        # detect events
        ed_parameters = self.default_parameters["event_detection"].copy()
        if detection_params is not None:
            for key, value in detection_params.items():
                assert key in ed_parameters, f"The parameter: {key} does not exist"
                ed_parameters[key] = value
    for input, output in zip(
        cellsets, self.get_results_filenames(f"{cellsetname}-ED", op=None)
    ):
        f_register["isx:event_detection"](input, output, ed_parameters)
        pb.update_progress_bar(1)

    if verbose:
        print("Event detection, done")

    # accept reject cells
    ar_parameters = self.default_parameters["accept_reject"].copy()
    if accept_reject_params is not None:
        for key, value in accept_reject_params.items():
            assert key in ar_parameters, f"The parameter: {key} does not exist"
            ar_parameters[key] = value

@register(['isx:auto_accept_reject'],"Accepting/Rejecting Cells")
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

@register('isx:multiple_registration',"Convining Planes",'MP')
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

            if overwrite:
                if os.path.exists(output_cell_set_file):
                    os.remove(output_cell_set_file)
                    json_file = json_filename(output_cell_set_file)
                    if os.path.exists(json_file):
                        os.remove(json_file)

            f_register["isx:multiplane_registration"](
                input_cell_set_files,
                ar_cell_set_file,
                output_cell_set_file,
                mpr_parameters,
                verbose,
            )
            f_register["isx:event_detection"](
                output_cell_set_file, ed_file, ed_parameters, verbose
            )
            f_register["isx:auto_accept_reject"](
                output_cell_set_file, ed_file, ar_cell_set_file, ar_parameters, verbose
            )
            self.output_file_paths.append(output_cell_set_file)
            pb.update_progress_bar(increment=1)
        print("done")

    def run_deconvolution(
        self,
        overwrite: bool = False,
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
            f_register["iisx:deconvolve_cellset"](
                cellset, denoise_file, ed_file, parameters
            )
            pb.update_progress_bar(increment=1)
        print("done")