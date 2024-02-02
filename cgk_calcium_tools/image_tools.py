import numpy as np
import matplotlib.pyplot as plt
import isx 

def plot_grid_max_dff_and_cellmap_fh(
    fh,
    ncols=5,
    eqhist=True,
    cellsetname="pca-ica",
    status_list=["accepted", "rejected", "undecided"],
    colors=['red','green', 'black'],
    single_plane:bool=True, **figure_kwargs):
    fig = plt.figure(**figure_kwargs)
    
    if single_plane:
        nfigures = len(fh.p_rec_paths)
        labels = fh.p_recording_labels
    else:
        nfigures = len(fh.rec_paths)
        labels = fh.recording_labels

    nrows = np.ceil(nfigures/ncols).astype(int)
    for i in range(nfigures):
        plt.subplot(nrows,ncols, i+1)
        plot_max_dff_and_cellmap_fh(
            fh, idx=i,
            eqhist=eqhist, cellsetname=cellsetname,
            status_list=status_list, colors=colors,
            op=None, single_plane=single_plane
        )
        plt.title(labels[i])
    plt.tight_layout()
    
def plot_max_dff_and_cellmap_fh(
    files_handler,
    idx=0,
    eqhist=True,
    cellsetname="pca-ica",
    status_list=["accepted", "rejected", "undecided"],
    colors=['red','green', 'black'],
    op=None,
    single_plane:bool=True
):
    """
    This function plots the maxdff figure (with an optional transformation) and adds the "borders" of the cell map

    Parameters
    ----------
    files_handler : isx_files_handler
        isx_files_handler to use.
    idx : int
        index of the session on files_handler.
    eqhist : bool
        Equalize histogram to hadnle extreme values and get a normalized distribution of intensities. Default True
    status_list : list
        Status of the cells to plot. Default: ['accepted', 'rejected', 'undecided']
    colors : list
        Colors of each status. Default: ['red','green', 'black']

    """
    if not single_plane:
        dataset = files_handler.get_results_filenames(cellsetname, op=op, idx=[idx], single_plane=False)[0]
        #choose the plane to show, the closest to the mean efocus
        media = np.mean(files_handler.efocus[idx])
        local_efocus_idx = np.argmin(np.abs(np.array(files_handler.efocus[idx]) - media)) 
        efocus_file = files_handler.focus_files[files_handler.rec_paths[idx]][local_efocus_idx]
        efocus_idx = files_handler.p_rec_paths.index(efocus_file)
        maxdff = files_handler.get_results_filenames("maxdff", op=op, idx=[efocus_idx], single_plane=True)[0]
    else:
        dataset = files_handler.get_results_filenames(cellsetname, op=op, idx=[idx], single_plane=True)[0]
        maxdff = files_handler.get_results_filenames("maxdff", op=op, idx=[idx], single_plane=True)[0]
    plot_max_dff_and_cellmap(dataset, maxdff, eqhist, status_list,colors=colors)


def plot_max_dff_and_cellmap(
    cellsetfile, maxdff, eqhist=True, status_list=["accepted", "rejected", "undecided"],colors=['red','green', 'black']
):
    """
    This function plots the maxdff figure (with an optional transformation) and add the "borders" of the cell map
    Parameters
    ----------
    cellsetfile : str
        file to load
    maxdff : str
        maxdff file.
    eqhist : bool
        Equalize histogram to handle extreme values and get a normalized distribution of intensities. Default True
    status_list : list
        Status of the cells to plot. Default: ['accepted', 'rejected', 'undecided']
    colors : list
        Colors of each status. Default: ['red','green', 'black']
    """
    assert len(colors)>=len(status_list), 'More colors are needed as the input'
    cell_set = isx.CellSet.read(cellsetfile)
    num_cells = cell_set.num_cells

    image_size = cell_set.get_cell_image_data(0).astype(np.float64)

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

def eqhist_tf(im, nbins=256):
    """
    This function transforms the value of the pixels of the image into having an uniform distribution of intensities
    """
    imhist, bins = np.histogram(im.flatten(), nbins, density=True)
    cdf = imhist.cumsum()
    cdf = 255 * cdf / cdf[-1]  # normalize

    # interpolate cdf to get thr new pixels values
    im2 = np.interp(im.flatten(), bins[:-1], cdf)

    return im2.reshape(im.shape)
