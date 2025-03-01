try:
    import isx
    isx_support = True
except ImportError:
    isx_support = False
    pass
import numpy as np
import os
from typing import Iterable
import warnings
from pathlib import Path
import shutil
from .files_io import RecordingFile, register_reader
from .recordings_handler import RecordingHandler
import re
from .jupyter_outputs import progress_bar
import json


def load_isxd_files( main_data_folder: str = ".",
    output_folder: str = ".",
    label: str = 'pipeline',
    files_pattern: str = r".*\.isxd",
    infere_hierarchy: bool = False,
    overwrite: bool = False):
    """
    This class helps handle and process Inscopix files/movies.

    Parameters
    ----------
    main_data_folder : str, optional
        Root folder containing the data. Output data follows
        the folder structure after this root folder. By default "."
    output_folder : str, optional
        Folder where the outputs are saved. By default "."        
    files_pattern : str, optional
        Naming patterns for the files. Also an easy way to select for one or
        multiple files from the same folder. By default "**/*.isx" (selects all)
    """
    regex = re.compile(files_pattern)

    files = []
    for file in Path(main_data_folder).rglob("*"):
        sfile = str(file)
        if file.is_file() and regex.match(sfile):                
            files.append(sfile) 
    
    recordings = []
    # Error if no files are found, lists if files were skipped
    assert (len(files) > 0), "No file(s) found"

    # Prints confirmation of number of files found and skipped.
    print(f"{len(files)} file(s) found.")

    # Initializes progress bar
    pb = progress_bar(len(files), "Loading")

    for file in files:
        # skip processing if metadata file already exists and overwrite is not allowed
        folder_structure = os.path.dirname(file)
        if folder_structure != '.':
            folder_structure = folder_structure[len(main_data_folder) + 1 :]


        json_file = os.path.join(
            output_folder,
            folder_structure,
            os.path.splitext(os.path.basename(file))[0] + f"_{label}_metadata.json",
        )
        if os.path.exists(json_file):
            if overwrite:
                os.remove(json_file)
            else:
                with open(json_file, "r") as file_data:
                    json_data = json.load(file_data)
                recordings.append(RecordingFile(**json_data))
                pb.update_progress_bar(1)
                continue

        # Read the video file and grab metadata information
        assert file.endswith(".isxd"), "Files must be inscopix's recordings"
        rec = read_isxd_file(main_data_folder,file)
        
        recordings.append(rec)

        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        with open(json_file, "w") as j_file:
            json.dump(rec._asdict(), j_file)
        # Update progress bar
        pb.update_progress_bar(1)

    return RecordingHandler(main_data_folder, recordings, output_folder, label,infere_hierarchy=infere_hierarchy)


@register_reader(".isxd")
def read_isxd_file(main_folder, file, output_folder='.'):
    video = isx.Movie.read(os.path.join(main_folder,file))
    resolution = video.spacing.num_pixels
    duration = video.timing.num_samples * video.timing.period.to_usecs() / 1e6
    frames_per_second = 1 / (video.timing.period.to_usecs() / 1e6)

    efocus = get_efocus(file, output_folder, video)
    video.flush()
    del video  # usefull for windows


    file_rel = file[len(main_folder) + 1:]
    return RecordingFile(
        file=file_rel,
        main_folder=main_folder,
        efocus=efocus,
        resolution=resolution,
        duration=duration,
        frames_per_second=frames_per_second,
        update_timestamp=[None],
        source_files=[None],
        creation_function={}
    )

def get_efocus(file,outputfolder,video):
# Lookig for multiplanes:
    raw_gpio_file = (
        os.path.splitext(file)[0] + ".gpio"
    )  
    # raw data for gpio
    updated_gpio_file = (
        os.path.splitext(file)[0] + "_gpio.isxd"
    )  # after the first reading gpio is converted to this
    local_updated_gpio_file = os.path.join(
        outputfolder,
        Path(updated_gpio_file).name,
    )  # new gpio copied in output
    if os.path.exists(local_updated_gpio_file):
        efocus = get_efocus_from_gpio(local_updated_gpio_file)
    elif os.path.exists(updated_gpio_file):
        efocus = get_efocus_from_gpio(updated_gpio_file)
    elif os.path.exists(raw_gpio_file):
        local_raw_gpio_file = os.path.join(
            outputfolder,
            Path(raw_gpio_file).name,
        )
        os.makedirs(os.path.dirname(local_raw_gpio_file), exist_ok=True)
        shutil.copy2(raw_gpio_file, local_raw_gpio_file)
        efocus = get_efocus_from_gpio(local_raw_gpio_file)
    else:
        get_acquisition_info = video.get_acquisition_info()
        if get_acquisition_info is not None and "Microscope Focus" in get_acquisition_info:
            if not isx.verify_deinterleave(
                file, get_acquisition_info["Microscope Focus"]
            ):
                warnings.warn(
                    f"Info {file}: Multiple Microscope Focus but not gpio file",
                    Warning,
                )
                efocus = [0]
            else:
                efocus = [get_acquisition_info["Microscope Focus"]]
        else:
            efocus = [0]
            print(
                f"Info: Unable to verify Microscope Focus config in: {file}"
            )
    return efocus



def get_efocus_from_gpio(gpio_file: str) -> list:
    """
    Read the gpio set from a file and get the data associated.

    Parameters
    ----------
    gpio_file : str
        path of gpio file

    Returns
    -------
    list
        list with the video_efocus

    """
    gpio_set = isx.GpioSet.read(gpio_file)
    efocus_values = gpio_set.get_channel_data(gpio_set.channel_dict["e-focus"])[1]
    efocus_values, efocus_counts = np.unique(efocus_values, return_counts=True)
    min_frames_per_efocus = 100
    video_efocus = efocus_values[efocus_counts >= min_frames_per_efocus]
    assert (
        video_efocus.shape[0] < 4
    ), f"{gpio_file}: Too many efocus detected, early frames issue."
    return [int(v) for v in video_efocus]

def create_random_movie(output_file: str,num_samples:int=1000,num_pixels:tuple=(300, 400)):

    timing = isx.Timing(num_samples=num_samples)
    spacing = isx.Spacing(num_pixels=num_pixels)
    movie = isx.Movie.write(output_file, timing, 
                            spacing, np.float32)
    
    for i in range(num_samples):
        movie.set_frame_data(i, np.random.random(num_pixels).astype(np.float32))
    movie.flush()
                                                                

def create_similar_empty_events(cell_set_file,ed_file):
    """
    Creates an empty event set file with the timing information from a given cell set file.

    Parameters:
        cell_set_file (str): The path to the cell set file.
        ed_file (str): The path to the event set file to be created.

    Returns:
        None
    """
    cell_set = isx.CellSet.read(cell_set_file)
    evset = isx.EventSet.write(ed_file, cell_set.timing, [""])
    evset.flush()
    del evset  # isx keeps the file open otherwise


def create_empty_cellset(output_cell_set_file: str,
                         pixels: tuple=(500, 300),
                         num_samples:int=3000,
                         msec:int=1000):
    
    timing = isx.Timing(num_samples=num_samples, period=isx.Duration.from_msecs(msec))
    spacing = isx.Spacing(num_pixels=pixels)
    cell_set = isx.CellSet.write(
        output_cell_set_file,
        timing,
        spacing,
    )
    image_null = np.zeros(spacing.num_pixels, dtype=np.float32)
    trace_null = np.zeros(timing.num_samples, dtype=np.float32)
    cell_set.set_cell_data(0, image_null, trace_null, "")
    cell_set.flush()
    del cell_set  # isx keeps the file open otherwise

def create_similar_empty_cellset(input_file: str, output_cell_set_file: str):
    """
    Creates a cellset file without cells with the features of the input file.

    Parameters:
        input_file (str): The path to the input cell set file.
        output_cell_set_file (str): The path to the output cell set file.

    Returns:
        None
    """
    cell_set_plane = isx.CellSet.read(input_file)
    cell_set = isx.CellSet.write(
        output_cell_set_file,
        cell_set_plane.timing,
        cell_set_plane.spacing,
    )
    image_null = np.zeros(cell_set.spacing.num_pixels, dtype=np.float32)
    trace_null = np.zeros(cell_set.timing.num_samples, dtype=np.float32)
    cell_set.set_cell_data(0, image_null, trace_null, "")
    cell_set.flush()
    del cell_set  # isx keeps the file open otherwise

def cellset_is_empty(cellset: str, accepted_only:bool = True):
    """
    Checks if a cellset file is empty.

    Args:
        cellset (str): The path to the cellset file.
        accepted_only (bool): If True, only consider accepted cells. Defaults to True.

    Returns:
        bool: True if the cellset is empty, False otherwise.
    """
    cs = isx.CellSet.read(cellset)
    is_empty = True
    
    if not accepted_only:
        is_empty = cs.num_cells == 0
    else:
        for n in range(cs.num_cells):
            if cs.get_cell_status(n) == "accepted":
                is_empty = False
                break
                
    cs.flush()
    del cs  # isx keeps the file open otherwise
    
    return is_empty


def get_segment_from_movie(
    inputfile: str,
    outputfile: str,
    borders: Iterable,
    keep_start_time=False,
    unit="minutes",
) -> None:
    """
    This function gets a segment of a video to create a new one. It's just an easy
    handler of isx.trim_movie, which trim frames from a movie to produce a new movie

    Parameters
    ----------
    inputfile : str
        Path of the input file to use
    outputfile : str
        Path of the output file to use
    borders: iterable
        With two elements of the borders in minutes (integers) of the segment.
        Where negative times means from the end like indexing in a numpy array
    keep_start_time : Bool, optional
        If true, keep the start time of the movie, even if some of its
        initial frames are to be trimmed, by default False
    unit: str
        the unit to use for border. It could be 'minutes' or 'seconds'

    Returns
    -------
    None

    Examples
    --------
    >>> get_segment_from_movie('inputfile.isxd','output.isxd',[-30, -1]) #last 30 minutes

    """
    assert len(borders) == 2, "borders must have two elements"
    assert unit in ["minutes", "seconds"], """unit could be 'minutes' or 'seconds'. """
    movie = isx.Movie.read(inputfile)
    if unit == "minutes":
        unitfactor = 60
    else:
        unitfactor = 1
    numframes = movie.timing.num_samples
    duration_unit = numframes * movie.timing.period.to_msecs() / (1000 * unitfactor)
    timing_period_unit = movie.timing.period.to_msecs() / (1000 * unitfactor)

    movie.flush()

    crop_segments = []
    assert (
        borders[0] > -duration_unit
    ), "asking for a segment from {minutes[0]} {unit} before the end, but video is {duration_minutes} {unit} long."
    assert (
        borders[0] < duration_unit
    ), "asking for a segment from {minutes[0]} {unit}, but video is {duration_minutes} {unit} long."
    assert (
        borders[1] < duration_unit
    ), "asking for a segment up to {minutes[0]} {unit}, but video is {duration_minutes} {unit} long."
    assert (
        borders[1] > -duration_unit
    ), "asking for a segment up to {minutes[0]} {unit} before the end, but video is {duration_minutes} {unit} long."

    # remove fist frames:
    # don't cut if the segments are from the beggining or just exactlt
    # duration before the end
    if borders[0] != 0 and borders[0] != -duration_unit:
        if borders[0] < 0:
            end1 = (duration_unit + borders[0]) / timing_period_unit - 1
        else:
            end1 = borders[0] / timing_period_unit - 1
        crop_segments.append([0, int(end1)])
    # remove last frames:
    # and don't cut if the segments are up to the end or just the exact
    # duration
    if borders[1] != -1 and borders[1] != duration_unit:
        if borders[1] < 0:
            start1 = (duration_unit + borders[1] + 1) / timing_period_unit + 1
        else:
            start1 = borders[1] / timing_period_unit + 1
        crop_segments.append([int(start1), numframes])

    if os.path.exists(outputfile):
        os.remove(outputfile)
    if len(crop_segments) == 0:
        print("no trim need it")
        return
    isx.trim_movie(
        input_movie_file=inputfile,
        output_movie_file=outputfile,
        crop_segments=np.array(crop_segments),
        keep_start_time=keep_start_time,
    )
