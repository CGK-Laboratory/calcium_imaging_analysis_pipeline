import isx
import pandas as pd
from pathlib import Path
import numpy as np
import os
from .files_io import write_log_file, same_json_or_remove, parameters_for_isx

def compute_traces_corr(fh, cellsetname: str, verbose=False) -> pd.DataFrame:
    """
    This function runs the isx.cell_metrics function, which compute cell metrics
    for a given cell set and events combination

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
        a concatenates list with metrics


    """
    cell_set_files = fh.get_results_filenames(
        f"{cellsetname}", op=None, single_plane=False
    )
    corr_files = fh.get_results_filenames(
        f"{cellsetname}_corr.csv", op=None, single_plane=False
    )
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
                tr1 = cs.get_cell_trace(i)
                for j in range(i+1, num_cells):
                    tr2 = cs.get_cell_trace(j)
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



def ixs_cell_metrics(fh, cellsetname: str, verbose=False) -> pd.DataFrame:
    """
    This function runs the isx.cell_metrics function, which compute cell metrics
    for a given cell set and events combination

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
        a concatenates list with metrics


    """
    cell_set_files = fh.get_results_filenames(
        f"{cellsetname}", op=None, single_plane=False
    )
    ed_files = fh.get_results_filenames(
        f"{cellsetname}-ED", op=None, single_plane=False
    )
    metrics_files = fh.get_results_filenames(
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
        metrics_files, fh.recording_labels, cell_set_files
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


def get_eventset(fh, cellsetname="pca-ica", accepted_only=True):

    event_det_files = fh.get_results_filenames(
        f"{cellsetname}-ED", op=None, single_plane=False
    )
    cellset_files = fh.get_results_filenames(
        f"{cellsetname}", op=None, single_plane=False
    )
    labels = fh.recording_labels
    events_df = []
    for events, cellset, label in zip(event_det_files, cellset_files, labels):
        ev = isx.EventSet.read(events)
        cs = isx.CellSet.read(cellset)
        for n in range(cs.num_cells):
            if not accepted_only or cs.get_cell_status(n) == "accepted":
                events_df.append(
                    pd.DataFrame(
                        {
                            "Recording Label": label,
                            "ed_file": events,
                            "Time (us)": ev.get_cell_data(n)[0],
                            "Amplitude": ev.get_cell_data(n)[1],
                            "cell_name": ev.get_cell_name(n),
                            "cell_status": cs.get_cell_status(n),
                        }
                    )
                )
    return pd.concat(events_df)
