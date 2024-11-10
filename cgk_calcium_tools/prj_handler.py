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

from .analysis_utils import apply_reject_criteria
from typing import Union
from .isxp_reader import get_parent_and_file

class isx_prj_handler:
    def __init__(self, main_folder:str, filter:str='', events_name:str='Events'):
        self.projects = []
        self.cellsets = []
        self.events = []
        manual_isx_files = glob.glob(f"{main_folder}/**/*.isxp", recursive=True)

        for isxp_file in manual_isx_files:
            if filter not in isxp_file[len(main_folder)+1:]:
                continue
            with open(isxp_file, "r") as f:
                s = f.read()
            try:
                parsed_project=json.loads(s)
            except:
                parsed_project=json.loads(s[0:-1])
            res= get_parent_and_file(parsed_project,'name', events_name)
            if res in None:
                continue
            self.projects.append(isxp_file)
            self.cellsets.append(os.path.join(os.path.dirname(isxp_file),res[0]))
            self.events.append(os.path.join(os.path.dirname(isxp_file),res[1]))
    
    def create_outputname(self, ending:str, from_cellset:bool=False) -> str:
        out = []
        if from_cellset:
            iter = self.cellsets
        else:
            iter = self.events
        for f in iter:
            out.append(f.with_suffix("")+'ending')
        return out
    
    def apply_reject_criteria(self, verbose=False) -> pd.DataFrame:
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
        status_files = self.create_outputname("_status.csv", from_cellset=False)
        return apply_reject_criteria(self.cellsets, metric_files, status_files, verbose=verbose)

     
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

        return apply_reject_criteria(self.cellsets, self.events, metric_files, verbose=verbose)


    