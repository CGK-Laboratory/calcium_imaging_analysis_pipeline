import numpy as np
import matplotlib.pyplot as plt
import isx
from typing import Union


def plot_grid_max_dff_and_cellmap_fh(
    fh,
    ncols=5,
    eqhist=True,
    cellsetname="pca-ica",
    status_list=["accepted", "rejected", "undecided"],
    colors=["red", "green", "black"],
    single_plane: bool = True,
    **figure_kwargs,
):
    plt.figure(**figure_kwargs)

    if single_plane:
        nfigures = len(fh.p_rec_paths)
        labels = fh.p_recording_labels
    else:
        nfigures = len(fh.rec_paths)
        labels = fh.recording_labels

    nrows = np.ceil(nfigures / ncols).astype(int)
    for i in range(nfigures):
        plt.subplot(nrows, ncols, i + 1)
        plot_max_dff_and_cellmap_fh(
            fh,
            idx=i,
            eqhist=eqhist,
            cellsetname=cellsetname,
            status_list=status_list,
            colors=colors,
            op=None,
            single_plane=single_plane,
        )
        plt.title(labels[i])
    plt.tight_layout()


def plot_max_dff_and_cellmap_fh(
    RecordingHandler,
    idx=0,
    eqhist=True,
    cellsetname="pca-ica",
    status_list=["accepted", "rejected", "undecided"],
    colors=["red", "green", "black"],
    op: Union[None, str] = None,
    single_plane: bool = True,
) -> None:
    """
    This function plots the maxdff figure (with an optional transformation) and adds the "borders" of the cell map

    Parameters
    ----------
    RecordingHandler : isx_RecordingHandler
        isx_RecordingHandler to use.
    idx : int, optional
        index of the session on RecordingHandler, by default 0.
    eqhist : bool, optional
        Equalize histogram to hadnle extreme values and get a normalized distribution of intensities. Default True
    cellsetname : str, optional
        refers to the name of the cell set, by default "pca-ica"
    status_list : list
        Status of the cells to plot. Default: ['accepted', 'rejected', 'undecided']
    colors : list
        Colors of each status. Default: ['red','green', 'black']
    op : str or None, optional
        Operation in the steps. If None, it will be the final one. Default: None
    single_plane : bool, optional
        If true, it returns the name for each plane; otherwise, it returns one
        string per recording. Default "True".

    Returns
    -------
    None

    """
    if not single_plane:
        dataset = RecordingHandler.get_results_filenames(
            cellsetname, op=op, idx=[idx], single_plane=False
        )[0]
        # choose the plane to show, the closest to the mean efocus
        media = np.mean(RecordingHandler.efocus[idx])
        local_efocus_idx = np.argmin(
            np.abs(np.array(RecordingHandler.efocus[idx]) - media)
        )
        efocus_file = RecordingHandler.focus_files[RecordingHandler.rec_paths[idx]][
            local_efocus_idx
        ]
        efocus_idx = RecordingHandler.p_rec_paths.index(efocus_file)
        maxdff = RecordingHandler.get_results_filenames(
            "maxdff", op=op, idx=[efocus_idx], single_plane=True
        )[0]
    else:
        dataset = RecordingHandler.get_results_filenames(
            cellsetname, op=op, idx=[idx], single_plane=True
        )[0]
        maxdff = RecordingHandler.get_results_filenames(
            "maxdff", op=op, idx=[idx], single_plane=True
        )[0]
    plot_max_dff_and_cellmap(dataset, maxdff, eqhist, status_list, colors=colors)


def plot_max_dff_and_cellmap(
    cellsetfile,
    maxdff,
    eqhist=True,
    status_list=["accepted", "rejected", "undecided"],
    colors=["red", "green", "black"],
) -> None:
    """
    This function plots the maxdff figure (with an optional transformation) and add the "borders" of the cell map
    Parameters
    ----------
    cellsetfile : str
        file to load
    maxdff : str
        maxdff file.
    eqhist : bool, optional
        Equalize histogram to handle extreme values and get a normalized distribution of
        intensities. Default True
    status_list : list, optional
        Status of the cells to plot. Default: ['accepted', 'rejected', 'undecided']
    colors : list, optional
        Colors of each status. Default: ['red','green', 'black']

    Returns
    -------
    None
    """
    assert len(colors) >= len(status_list), "More colors are needed as the input"
    cell_set = isx.CellSet.read(cellsetfile)
    num_cells = cell_set.num_cells

    image = isx.Image.read(maxdff).get_data()
    if eqhist:
        image = eqhist_tf(image)

    plt.imshow(image, "Blues", interpolation="none", resample=False)
    for cell in range(num_cells):
        if cell_set.get_cell_status(cell) in status_list:
            cindex = status_list.index(cell_set.get_cell_status(cell))
            cell_image = cell_set.get_cell_image_data(cell).astype(np.float64)
            plt.contour(
                cell_image > (0.8 * np.max(cell_image)),
                colors=colors[cindex],
                linewidths=[2],
                alpha=0.3,
            )

    cell_set.flush()
    plt.xticks([])
    plt.yticks([])


def eqhist_tf(im: np.ndarray, nbins: int = 256) -> np.ndarray:
    """
    This function transforms the value of the pixels of the image into having an uniform
    distribution of intensities

    Parameters
    ----------
    im : np.ndarray
        the image
    nbins : int
        number of bins for the initial histogram
    Returns
    -------
    np.ndarray
        the image with the uniform distribution
    """
    imhist, bins = np.histogram(im.flatten(), nbins, density=True)
    cdf = imhist.cumsum()
    cdf = 255 * cdf / cdf[-1]  # normalize

    # interpolate cdf to get thr new pixels values
    im2 = np.interp(im.flatten(), bins[:-1], cdf)

    return im2.reshape(im.shape)
