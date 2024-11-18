import os
from pathlib import Path
from glob import glob
import warnings
import numpy as np
import isx
import json
from typing import Union, Tuple

import shutil
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

from .analysis_utils import apply_quality_criteria, compute_metrics
from typing import Union
from .isxp_reader import get_parent_and_file


class isx_prj_handler:
    def __init__(self, main_folder: str, str_filter: str = "", events_name: str = "Events"):
        self.projects = []
        self.cellsets = []
        self.events = []
        manual_isx_files = glob(f"{main_folder}/**/*.isxp", recursive=True)

        for isxp_file in manual_isx_files:
            if str_filter not in isxp_file[len(main_folder) + 1 :]:
                continue
            with open(isxp_file, "r") as f:
                s = f.read()
            try:
                parsed_project = json.loads(s)
            except:
                parsed_project = json.loads(s[0:-1])
            res = get_parent_and_file(parsed_project, "name", events_name)
            if res is None:
                continue
            self.projects.append(isxp_file)
            cellsetfile = os.path.join(os.path.dirname(isxp_file), res[0])
            eventsfile = os.path.join(os.path.dirname(isxp_file), res[1])
            if not os.path.exists(cellsetfile):
                warnings.warn(f"Warning: File {cellsetfile} not exists, omitted ", UserWarning)
                continue
            if not os.path.exists(eventsfile):
                warnings.warn(f"Warning: File {eventsfile} not exists, omitted ", UserWarning)
                continue
            self.cellsets.append(cellsetfile)
            self.events.append(eventsfile)

    def get_status(self):
        data = []
        status_files = self.create_outputname("_status.csv", from_cellset=False)
        for events,statusf in zip(self.events,status_files):
            df = pd.read_csv(statusf, index_col=0)
            df['File'] = events
            data.append(df)
        return pd.concat(data)
    
    def get_events(self, cells_used="accepted"):
        assert cells_used in ["accepted", "isx_accepted", "all"]
        data = []

        for cellset, events in zip(self.cellsets, self.events):
            if cells_used == "accepted":
                status_file = os.path.splitext(events)[0] + "_status.csv"
                status = pd.read_csv(status_file, index_col=0)               
            cs = isx.CellSet.read(cellset)
            es = isx.EventSet.read(events)
            cs_names = [cs.get_cell_name(i) for i in range(cs.num_cells)]
            for c in range(es.num_cells):
                cellname = es.get_cell_name(c)
                if (
                    cells_used == "isx_accepted"
                    and cs.get_cell_status(cs_names.index(cellname)) != "accepted"
                ):
                    continue
                elif (
                    cells_used == "accepted" and not status.loc[cellname, "corr_accepted"] or not status.loc[cellname, 'skew_accepted']
                ):
                    continue
                data.append(
                    {
                        "Events (s)": es.get_cell_data(c)[0] / 1e6,
                        "File": events,
                        "Duration (s)": cs.timing.num_samples
                        * cs.timing.period.to_usecs()
                        / 1e6,
                        "cell_name": cellname,
                    }
                )

    def create_outputname(self, ending: str, from_cellset: bool = False) -> str:

        out = []
        if from_cellset:
            iter = self.cellsets
        else:
            iter = self.events
        for f in iter:
            out.append(os.path.splitext(f)[0] + ending)
        return out

    def apply_quality_criteria(
        self,
        max_corr=0.9,
        min_skew=0.05,
        only_isx_accepted=True,
        overwrite=False,
        verbose=False,
    ) -> pd.DataFrame:
        """

        Parameters
        ----------
        verbose : bool, optional
            Show additional messages, by default False

        Returns
        -------
        pd.DataFrame
            DataFrame with the correlation matrix
        """
        metric_files = self.create_outputname("_metrics.csv", from_cellset=False)
        status_files = self.create_outputname("_status.csv", from_cellset=False)
        return apply_quality_criteria(
            self.cellsets,
            metric_files,
            status_files,
            max_corr=max_corr,
            min_skew=min_skew,
            only_isx_accepted=only_isx_accepted,
            overwrite=overwrite,
            verbose=verbose,
        )

    def compute_metrics(self, verbose=False) -> pd.DataFrame:
        """
        This function compute the  correlation matrix for the cell traces.

        Parameters
        ----------
        verbose : bool, optional
            Show additional messages, by default False

        Returns
        -------
        pd.DataFrame
            DataFrame with the correlation matrix
        """

        metric_files = self.create_outputname("_metrics.csv", from_cellset=False)

        return compute_metrics(
            self.cellsets, self.events, metric_files, verbose=verbose
        )
