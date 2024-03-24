import isx
import pandas as pd
from pathlib import Path
import numpy as np

def cellmetrics(fh, cellsetname="pca-ica"):
    ev = isx.EventSet.read()
    cs = isx.CellSet.read()


def compute_events_rates(events_df,cellsets_df):    
    fr = events_df.groupby(["Recording Label", "cell_name"])["Time (us)"].count().rename('#events').reset_index()  
    
    fr['Events per minute'] = fr.apply(lambda row: 60*row['#events']/cellsets_df.loc[row["Recording Label"],"Duration (s)"], axis=1)
    return fr


def compute_acumulated(events_df, cellsets_df, datapoints_per_minute=2, normalized=False):
    
    cum_events_df = []
    for ed_file, df in events_df.groupby('ed_file'):
        label = df["Recording Label"].values[0]
        maxtime_minutes = cellsets_df.loc[label,"Duration (s)"] /60
        time_axis = np.linspace(0, maxtime_minutes, int(maxtime_minutes * datapoints_per_minute))
        for cell in df["cell_name"].unique():
            evs_minutes = np.sort(
                df[(df["cell_name"] == cell)]["Time (us)"].values
            )/(60*1000**2)
            calc_cum_events = np.vectorize(lambda x: sum(evs_minutes < x))
            cum_events = calc_cum_events(time_axis)
            aux_df = pd.DataFrame({
                                "ed_file": ed_file,
                                "Recording Label": label,
                                "cell_name": cell,
                                "Time (minutes)": time_axis,
                                "Cumulative Events": cum_events.copy(),
                      })
            aux_df["Cumulative Distribution Function"]= cum_events.copy()/np.max(cum_events)
            cum_events_df.append(aux_df)
            
    return pd.concat(cum_events_df)

def get_movies_info(rec_paths)->pd.DataFrame:
    video_data = []
    for file in rec_paths:
        movie = isx.Movie.read(file)
        video_data.append(
            {
                "Resolution": movie.spacing.num_pixels,
                "Duration (s)": movie.timing.num_samples * 
                    movie.timing.period.to_usecs() / 1000000,
                "Sampling Rate (Hz)": 1 / (movie.timing.period.to_usecs() / 1000000),
                "Full Path": Path(file),
                "Main Name": str(Path(file).name),
                "Subfolder": str(Path(file).parent.name),
            }
        )
        movie.flush()
        del movie
    return pd.DataFrame(video_data)

def get_truncated_motion_correction_prop(fh, prop_thr = 0.05, translation_thr= None,op='MC'):
    translation_files = fh.get_results_filenames("translations.csv",op=op)
    if translation_thr is None:
        translation_thr = fh.default_parameters['motion_correct']['max_translation']

    for f in translation_files:
        translation_files_df = pd.read_csv(f)
        arr = np.logical_or(translation_files_df.translationY.abs()==translation_thr,
                                translation_files_df.translationX.abs()==translation_thr)
        prop = float(sum(arr))/len(arr)
        if  prop > prop_thr:
            print('rate of {:.2f} of frames with truncated motion corrextion in file {}'.format(prop,f))
        


def get_cellset_info(fh, cellset_name="pca-ica")->pd.DataFrame:
    cellset_files = fh.get_results_filenames(f"{cellset_name}", op=None,single_plane = False)
    labels = fh.recording_labels
    cellset_data = []
    for file, label in zip(cellset_files,labels):
        cellset = isx.CellSet.read(file)
        cellset_data.append(
            {
                "Recording Label": label,
                "Resolution": cellset.spacing.num_pixels,
                "Duration (s)": cellset.timing.num_samples * 
                    cellset.timing.period.to_msecs() / 1000,
                "Sampling Rate (Hz)": 1 / (cellset.timing.period.to_msecs() / 1000),
                "Full Path": Path(file),
                "Main Name": str(Path(file).name),
                "Subfolder": str(Path(file).parent.name),
            }
        )
    return pd.DataFrame(cellset_data).set_index("Recording Label")




def get_eventset(fh, cellsetname="pca-ica", accepted_only = True):

    event_det_files = fh.get_results_filenames(f"{cellsetname}-ED", op=None,single_plane = False)
    cellset_files = fh.get_results_filenames(f"{cellsetname}", op=None,single_plane = False)
    labels = fh.recording_labels
    events_df = []
    for events,cellset, label in zip(event_det_files, cellset_files, labels):
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
                            "cell_status": cs.get_cell_status(n)
                        }
                    )
                )
    return  pd.concat(events_df)