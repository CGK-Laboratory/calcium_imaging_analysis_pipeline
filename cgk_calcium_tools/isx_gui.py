from .isx_aux_functions import isx_support
if isx_support is False:
    pass

import numpy as np
from pathlib import Path
import os
from .cellsets_handler import CellsetsHandler
import isx


def create_inscopix_projects(fh: CellsetsHandler, cellsetname="pca-ica"):
    """
    Creates an Inscopix project file (.isxp) from a given isx_RecordingHandler object and cellset result.

    Parameters:
    fh (isx_RecordingHandler): The isx_RecordingHandler object containing the necessary files and information.
    cellsetname (str, optional): The name of the cellset to use. Defaults to "pca-ica".

    Returns:
    None
    """

    src_dir = os.path.dirname(os.path.abspath(__file__))
    with open(
        os.path.join(src_dir, "prj_template.json"), "r", encoding="utf-8"
    ) as file:
        prj_template = file.read()
    with open(
        os.path.join(src_dir, "single_plane_template.json"), "r", encoding="utf-8"
    ) as file:
        single_plane_template = file.read()

    for main_file, single_planes in fh.focus_files.items():
        idxs = [fh.p_rec_paths.index(f) for f in single_planes]
        cellsets = fh.get_results_filenames(f"{cellsetname}", op=None, idx=idxs)
        evs_dets = fh.get_results_filenames(f"{cellsetname}-ED", op=None, idx=idxs)
        dffs = fh.get_results_filenames("dff", op="MC", idx=idxs)

        idx = [fh.rec_paths.index(main_file)]

        project_file = fh.get_results_filenames(
            f"{cellsetname}.isxp", op=None, idx=idx, single_plane=False
        )[0]
        data_folder = project_file[:-5] + "_data"
        os.makedirs(data_folder, exist_ok=True)
        single_planes_info = []
        for dff, cellset, ev_det in zip(dffs, cellsets, evs_dets):
            parsed_plane = single_plane_template
            movie = isx.Movie.read(dff)
            movie_data = movie.get_frame_data(0)
            dmin = np.min(movie_data)
            dmax = np.max(movie_data)
            del movie
            replacements = {
                "{eventdet_path}": ev_det.replace("\\", "/"),
                "{eventdet_name}": Path(ev_det).name,
                "{cellset_path}": cellset.replace("\\", "/"),
                "{cellset_name}": Path(cellset).name,
                "{DFF_path}": dff.replace("\\", "/"),
                "{DFF_name}": Path(dff).name,
                "{dmax}": str(dmax),
                "{dmin}": str(dmin),
            }

            for key, value in replacements.items():
                parsed_plane = parsed_plane.replace(key, value)

            single_planes_info.append(parsed_plane)

        prj_text = prj_template.replace("{prj_name}", Path(project_file).name).replace(
            "{plane_1ist}", ", ".join(single_planes_info)
        )
        with open(project_file, "wt") as file:
            file.write(prj_text)
