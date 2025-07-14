import isx
import pandas as pd
from pathlib import Path
import numpy as np
import os
from .files_io import write_log_file, same_json_or_remove, parameters_for_isx
from scipy.stats import skew

def apply_quality_criteria(cell_set_files: list, metric_files: list, status_files: list,
                          max_corr=0.9, min_skew = 0.05, only_isx_accepted=True,overwrite=False, verbose=False) -> None:
    
    """
    This function takes a list of isx cellset files, a list of csv files with metrics for each cell, and a list of csv files where the status of the cells will be stored.
    It will compute the correlation matrix for the traces of each cell and if the correlation is higher than max_corr with otherone with higher snr, the cell will be labeled as "rejected"
    The status of the cells is stored in the status_files as a csv file with two columns: cellName and status.
    The metrics for each cell is also stored in the status_files as a csv file with the same columns as the input metric_files.
    
    Parameters
    ----------
    cell_set_files : list
        List of isx cellset files.
    metric_files : list
        List of csv files with metrics for each cell.
    status_files : list
        List of csv files where the status of the cells will be stored.
    max_corr : float, optional
        Maximum correlation allowed between cells. If the correlation is higher than this value, the cell will be labeled as "rejected". Default 0.9
    min_skew : float, oprtional
        Minimum skewness allowed for the traces of the cells. If the skewness is lower than this value, the cell will be labeled as "rejected". Default 0.05
    only_isx_accepted : bool, optional
        If True, only the cells that are labeled as "accepted" in the isx cellset file will be used. Default True
    overwrite : bool, optional
        If True, the status_files will be overwritten if they exist. Default False
    verbose : bool, optional
        If True, additional messages will be printed. Default False

    """
    for cellset, metric_file,status_file in zip(cell_set_files, metric_files,status_files):
        inputs_args = {
            "input_cell_set_file": os.path.basename(cellset),
            "input_metric_file": os.path.basename(cellset),
            "output_status_file": os.path.basename(status_file),
            "max_corr":max_corr,
            "min_skew":min_skew
        }
        
        if overwrite and os.path.exists(status_file):
            os.remove(status_file)
        input_files = ["input_cell_set_file","input_metric_file"]
        if same_json_or_remove(
            inputs_args,
            output=status_file,
            verbose=verbose,
            input_files_keys=input_files,
            
        ):
            continue
        cs = isx.CellSet.read(cellset)
        metrics = pd.read_csv(metric_file)
        status = pd.DataFrame.from_dict(
            {
                cell: [cs.get_cell_name(cell), cs.get_cell_status(cell)]
                for cell in range(metrics.shape[0])
            },
            orient="index",
            columns=["cellName", "status"],
        )
        metrics = metrics.merge(status, on="cellName")
        if  only_isx_accepted:
            metrics = metrics[metrics["status"] == "accepted"]
        metrics=metrics.set_index('cellName')

        metrics['skew'] = np.nan
        if  only_isx_accepted:
            cells = np.array([i for i in range(cs.num_cells) if cs.get_cell_status(i) == "accepted" ])
            num_cells =len(cells)
        else:
            num_cells = cs.num_cells
            cells = np.arange(num_cells)
        
        #compute trace correlations
        corr_labels = [cs.get_cell_name(i) for i in cells]
        corr = np.zeros((num_cells, num_cells))
        for ix in range(num_cells):
            tr1 = cs.get_cell_trace_data(cells[ix])
            metrics.loc[corr_labels[ix],'skew'] = skew(tr1, nan_policy='omit')
            for jx in range(ix+1, num_cells):
                tr2 = cs.get_cell_trace_data(cells[jx])
                valid = (~np.isnan(tr1)) * (~np.isnan(tr2))
                corr[ix, jx] = np.corrcoef(tr1[valid], tr2[valid])[0, 1]
                corr[jx, ix] = corr[ix, jx]
                if np.isnan(corr[jx, ix]):
                    continue

        metrics['skew_accepted'] = metrics['skew'] >= min_skew

        cs.flush()
        hcorr=np.nonzero(corr>max_corr)
        
        metrics['corr_accepted'] = True
        metrics['corr_rejected_by'] = ''

def compute_traces_corr(cell_set_files: list, corr_files: list, verbose=False) -> pd.DataFrame:
    """
    This function compute the correlation matrix for the cell traces.

    Parameters
    ----------
    fh : isx_files_handler
        isx_files_handler object
    cellsetname : str
        cell label to get filename
    verbose : bool, optional
        Show additional messages, by default False

    Returns
    -------
    pd.DataFrame
        DataFrame with the correlation matrix
    """

    for cellset, corr_file in zip(cell_set_files, corr_files):
        inputs_args = {
            "input_cell_set_file": os.path.basename(cellset),
            "output_corr_file": os.path.basename(corr_file),
        }
        if not same_json_or_remove(
            inputs_args,
            output=corr_file,
            verbose=verbose,
            input_files_keys=["input_cell_set_file"],
        ):
            
            cs = isx.CellSet.read(cellset)
            num_cells = cs.num_cells
            corr = np.zeros((num_cells, num_cells))
            for i in range(num_cells-1):
                tr1 = cs.get_cell_trace_data(i)
                for j in range(i+1, num_cells):
                    tr2 = cs.get_cell_trace_data(j)
                    corr[i, j] = np.corrcoef(tr1, tr2)[0, 1]
                    corr[j, i] = corr[i, j]
            labels = [cs.get_cell_name(i) for i in range(cs.num_cells)]
            df = pd.DataFrame(corr, index=labels, columns=labels)
            df.to_csv(corr_file)
            write_log_file(
                inputs_args,
                os.path.dirname(corr_file),
                {"function": "cell_corr"},
                input_files_keys=["input_cell_set_file"],
                output_file_key="output_corr_file",
            )
    return



        good_cells = metrics[metrics['skew_accepted']].sort_values('snr', ascending=False).index
        for good_cell in good_cells:
            cell_ix = corr_labels.index(good_cell)
            if cell_ix in hcorr[0]:
                cells2rm_ix = hcorr[1][hcorr[0]==cell_ix]
                for cell2rm_ix in cells2rm_ix:
                    cell2rm = corr_labels[cell2rm_ix]
                    if metrics.loc[cell2rm,'corr_accepted']:
                        metrics.loc[cell2rm,'corr_accepted'] = False
                        metrics.loc[cell2rm,'corr_rejected_by'] = good_cell

        metrics['accepted'] = metrics['skew_accepted'] & metrics['corr_accepted']
        metrics.to_csv(status_file)
        write_log_file(
            inputs_args,
            os.path.dirname(status_file),
            {"function": "cgk_apply_reject"},
            input_files_keys=input_files,
            output_file_key="output_status_file",
        )
    
    return




def compute_metrics(cell_set_files: list, ed_files: list, metrics_files: list, verbose=False) -> pd.DataFrame:
    """
    This function runs the isx.cell_metrics function, which compute cell metrics
    for a given cell set and events combination

    Parameters
    ----------
    verbose : bool, optional
        Show additional messages, by default False

    Returns
    -------
    pd.DataFrame
        a concatenates list with metrics


    """

    for cellset, ed, metric in zip(cell_set_files, ed_files, metrics_files):
        inputs_args = {
            "input_cell_set_files": os.path.basename(cellset),
            "input_event_set_files": os.path.basename(ed),
            "output_metrics_file": os.path.basename(metric),
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
                            "output_metrics_file": metric,
                        },
                    )
                )
            except Exception as e:
                code_medtris = isx.cell_metrics.__code__
                if 'output_metrics_file' not in code_medtris.co_varnames[:code_medtris.co_argcount]: #patching for isx <1.9.4
                    isx.cell_metrics(
                        **parameters_for_isx(
                            inputs_args,
                            keys_to_remove=["output_metrics_file"],
                            to_update={
                                "input_cell_set_files": cellset,
                                "input_event_set_files": ed,
                                "output_metrics_files": metric,
                            },
                        )
                    )
                else:
                    raise(e)
            write_log_file(
                inputs_args,
                os.path.dirname(metric),
                {"function": "cell_metrics"},
                input_files_keys=["input_cell_set_files", "input_event_set_files"],
                output_file_key="output_metrics_file",
            )

    df = []
    for metric_file, cell_set_file in zip(
        metrics_files, cell_set_files
    ):
        aux = pd.read_csv(metric_file)
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


def compute_events_rates(events_df, cellsets_df):
    fr = (
        events_df.groupby(["Recording Label", "cell_name"])["Time (us)"]
        .count()
        .rename("#events")
        .reset_index()
    )

    fr["Events per minute"] = fr.apply(
        lambda row: 60
        * row["#events"]
        / cellsets_df.loc[row["Recording Label"], "Duration (s)"],
        axis=1,
    )
    return fr


def compute_acumulated(
    events_df, cellsets_df, datapoints_per_minute=2, normalized=False
):

    cum_events_df = []
    for ed_file, df in events_df.groupby("ed_file"):
        label = df["Recording Label"].values[0]
        maxtime_minutes = cellsets_df.loc[label, "Duration (s)"] / 60
        time_axis = np.linspace(
            0, maxtime_minutes, int(maxtime_minutes * datapoints_per_minute)
        )
        for cell in df["cell_name"].unique():
            evs_minutes = np.sort(df[(df["cell_name"] == cell)]["Time (us)"].values) / (
                60 * 1000**2
            )
            calc_cum_events = np.vectorize(lambda x: sum(evs_minutes < x))
            cum_events = calc_cum_events(time_axis)
            aux_df = pd.DataFrame(
                {
                    "ed_file": ed_file,
                    "Recording Label": label,
                    "cell_name": cell,
                    "Time (minutes)": time_axis,
                    "Cumulative Events": cum_events.copy(),
                }
            )
            aux_df["Cumulative Distribution Function"] = cum_events.copy() / np.max(
                cum_events
            )
            cum_events_df.append(aux_df)

    return pd.concat(cum_events_df)


def get_movies_info(rec_paths) -> pd.DataFrame:
    video_data = []
    for file in rec_paths:
        movie = isx.Movie.read(file)
        video_data.append(
            {
                "Resolution": movie.spacing.num_pixels,
                "Duration (s)": movie.timing.num_samples
                * movie.timing.period.to_usecs()
                / 1000000,
                "Sampling Rate (Hz)": 1 / (movie.timing.period.to_usecs() / 1000000),
                "Full Path": Path(file),
                "Main Name": str(Path(file).name),
                "Subfolder": str(Path(file).parent.name),
            }
        )
        movie.flush()
        del movie
    return pd.DataFrame(video_data)


def get_truncated_motion_correction_prop(
    fh, prop_thr=0.05, translation_thr=None, op="MC"
):
    translation_files = fh.get_results_filenames("translations.csv", op=op)
    if translation_thr is None:
        translation_thr = fh.default_parameters["motion_correct"]["max_translation"]

    for f in translation_files:
        translation_files_df = pd.read_csv(f)
        arr = np.logical_or(
            translation_files_df.translationY.abs() == translation_thr,
            translation_files_df.translationX.abs() == translation_thr,
        )
        prop = float(sum(arr)) / len(arr)
        if prop > prop_thr:
            print(
                "rate of {:.2f} of frames with truncated motion corrextion in file {}".format(
                    prop, f
                )
            )


def get_cellset_info(fh, cellset_name="pca-ica") -> pd.DataFrame:
    cellset_files = fh.get_results_filenames(
        f"{cellset_name}", op=None, single_plane=False
    )
    labels = fh.recording_labels
    cellset_data = []
    for file, label in zip(cellset_files, labels):
        cellset = isx.CellSet.read(file)
        cellset_data.append(
            {
                "Recording Label": label,
                "Resolution": cellset.spacing.num_pixels,
                "Duration (s)": cellset.timing.num_samples
                * cellset.timing.period.to_msecs()
                / 1000,
                "Sampling Rate (Hz)": 1 / (cellset.timing.period.to_msecs() / 1000),
                "Full Path": Path(file),
                "Main Name": str(Path(file).name),
                "Subfolder": str(Path(file).parent.name),
            }
        )
    return pd.DataFrame(cellset_data).set_index("Recording Label")

def get_events(cellset_files, event_det_files, cells_used="accepted"):
    assert cells_used in ["accepted", "isx_accepted", "all"]

    data = []
    for cellset, events in zip(cellset_files, event_det_files):
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
                cells_used == "accepted" and not status.loc[cellname, "accepted"]
            ):
                continue
            data.append(pd.DataFrame(
                {
                    "Events (s)": es.get_cell_data(c)[0] / 1e6,
                    "File": events,
                    "Duration (s)": cs.timing.num_samples
                    * cs.timing.period.to_usecs()
                    / 1e6,
                    "cell_name": cellname,
                }
            ))
    return pd.concat(data)
