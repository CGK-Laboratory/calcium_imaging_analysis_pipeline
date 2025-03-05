from .analysis_utils import apply_quality_criteria, compute_metrics, get_events
import pandas as pd
class CellsetsHandler:
    def apply_quality_criteria(
        self,
        cellsetname: str,
        max_corr=0.9,
        min_skew=0.05,
        only_isx_accepted=True,
        overwrite=False,
        verbose=False,
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
            cell_set_files,
            metrics_files,
            status_files,
            max_corr=max_corr,
            min_skew=min_skew,
            only_isx_accepted=only_isx_accepted,
            overwrite=overwrite,
            verbose=verbose,
        )

    def get_status(self, cellsetname: str):
        data = []
        status_files = self.get_results_filenames(
            f"{cellsetname}-ED_status.csv", op=None, single_plane=False
        )
        for events, statusf in zip(self.events, status_files):
            df = pd.read_csv(statusf, index_col=0)
            df["File"] = events
            data.append(df)
        return pd.concat(data)

    def get_events(self, cellsetname: str, cells_used="accepted"):

        event_det_files = self.get_results_filenames(
            f"{cellsetname}-ED", op=None, single_plane=False
        )
        cellset_files = self.get_results_filenames(
            f"{cellsetname}", op=None, single_plane=False
        )
        # TODO: it could be merge with recording_labels
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

    