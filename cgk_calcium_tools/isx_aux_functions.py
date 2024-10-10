import isx
import numpy as np


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