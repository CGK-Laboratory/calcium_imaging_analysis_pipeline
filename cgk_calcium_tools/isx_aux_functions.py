import isx




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
    n_input = cs.num_cells
    is_empty = True
    
    for n in range(n_input):
        if not accepted_only or cs.get_cell_status(n) == "accepted":
            is_empty = False
            break
                
    cs.flush()
    del cs  # isx keeps the file open otherwise
    
    return is_empty