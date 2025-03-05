import os
from glob import glob
import warnings
import json

import pandas as pd

from .analysis_utils import apply_quality_criteria, compute_metrics,get_events
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
            if s[-1]=='\x00':
                s = s[0:-1]
            parsed_project = json.loads(s)
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

        if len(self.cellsets) == 0:
            raise Exception("No isx files found")
    def get_status(self):
        data = []
        status_files = self.create_outputname("_status.csv", from_cellset=False)
        for events,statusf in zip(self.events,status_files):
            df = pd.read_csv(statusf, index_col=0)
            df['File'] = events
            data.append(df)
        return pd.concat(data)
    
    def get_events(self, cells_used="accepted"):
        return get_events(self.cellsets, self.events, cells_used=cells_used)

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
