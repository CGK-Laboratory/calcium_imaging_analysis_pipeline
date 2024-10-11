import isx
import numpy as np
import os
from typing import Iterable

def create_empty_events(cell_set_file,ed_file):
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

def create_empty_cellset(input_file: str, output_cell_set_file: str):
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
