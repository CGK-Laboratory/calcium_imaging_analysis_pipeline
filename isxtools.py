import os
from pathlib import Path
from glob import glob
import numpy as np
import isx
import matplotlib.pyplot as plt


class isx_files_handler:
    """
    This class helps to iterate over files for inscopix processing

    Parameters
    ----------
    main_data_folder : str, path or list
        Folder with data, after this path the output will share the folder structure. If it's a list each element should correspond with each data_subfolders and files_pattern.
    data_subfolders : list
        Where each file (or files) following each files_pattern element are.
    files_pattern : list
        Naming patterns for the files, and easy way to handle seleection one or multiple files from the same folder.
    outputsfolder : list
        Folder where the file structure (following only the data_subfolders) will be copy and the results written.
    processing_steps: list
        Naming steps will be use, adding one affter the previous ones.
    one_file_per_folder: bool
        If True it will check than one and only one file is found with the pattern in it's folder

    """

    def __init__(
        self,
        main_data_folder,
        data_subfolders,
        files_pattern,
        outputsfolder,
        processing_steps=["trim", "PP", "BP", "MC"],
        one_file_per_folder=True,
        recording_labels=None,
    ) -> None:
        self.outputsfolder = Path(outputsfolder)
        self.data_subfolders = data_subfolders
        if not os.path.exists(self.outputsfolder):
            os.mkdir(self.outputsfolder)
        for s in data_subfolders:
            os.makedirs(self.outputsfolder / s, exist_ok=True)
        rec_names = []
        for i, f in enumerate(data_subfolders):
            if isinstance(main_data_folder, list):
                this_main_data_folder = Path(main_data_folder[i])
            else:
                this_main_data_folder = Path(main_data_folder)
            files = glob(
                str(this_main_data_folder / f / files_pattern[i]), recursive=False
            )
            if one_file_per_folder:
                assert len(files) == 1, "None or multiple files found for {}.".format(
                    str(this_main_data_folder / f / files_pattern[i])
                )
                rec_names.append(files[0])
            else:
                assert len(files) > 0, "No file found for {}.".format(
                    str(this_main_data_folder / f / files_pattern[i])
                )
                [rec_names.append(f) for f in files if f not in rec_names]
        self.rec_names = rec_names
        self.processing_steps = processing_steps
        if recording_labels is None:
            self.recording_labels = rec_names
        else:
            self.recording_labels = recording_labels

    def get_pair_filenames(self, operation):
        """
        This method return the input/output pairs for the given operation step operation.
        Parameters
        ----------
        operation : str
            Operation in the steps.
        """
        assert (
            operation in self.processing_steps
        ), f"operation must be in the list {self.processing_steps}, got: {operation}"

        opi = self.processing_steps.index(operation)
        outputs = []
        inputs = []
        suffix_out = "-" + "-".join(self.processing_steps[: opi + 1]) + ".isxd"
        if opi == 0:
            suffix_in = None
            inputs = self.rec_names
        else:
            suffix_in = "-" + "-".join(self.processing_steps[:opi]) + ".isxd"

        for fname, file in zip(self.data_subfolders, self.rec_names):
            outputs.append(
                str(Path(self.outputsfolder, fname, Path(file).stem + suffix_out))
            )
            if suffix_in is not None:
                inputs.append(
                    str(Path(self.outputsfolder, fname, Path(file).stem + suffix_in))
                )

        return zip(inputs, outputs)

    def get_filenames(self, op=None):
        """
        This method return the filenames of the files create after the operation op.
        Parameters
        ----------
        op : str
            Operation in the steps. If None, it will be the final one. Default: None
        """
        if op is None:
            names = self.processing_steps
        else:
            assert (
                op in self.processing_steps
            ), f"op must be in the list {self.processing_steps}, got: {op}"
            opi = self.processing_steps.index(op)
            names = self.processing_steps[: opi + 1]
        suffix_out = "-" + "-".join(names) + ".isxd"
        outputs = []
        for fname, file in zip(self.data_subfolders, self.rec_names):
            outputs.append(
                str(Path(self.outputsfolder, fname, Path(file).stem + suffix_out))
            )
        return outputs

    def get_results_filenames(self, name, op=None, subfolder="", idx=None):
        if op is not None:
            opi = self.processing_steps.index(op)
            steps = self.processing_steps[: (opi + 1)]
        else:
            steps = self.processing_steps
        if "." in name:
            ext = ""
        else:
            ext = ".isxd"
        suffix_out = "-" + "-".join(steps) + "-{}{}".format(name, ext)
        outputs = []

        for i, (fname, file) in enumerate(zip(self.data_subfolders, self.rec_names)):
            if idx is None or i in idx:
                outputs.append(
                    str(
                        Path(
                            self.outputsfolder,
                            fname,
                            subfolder,
                            Path(file).stem + suffix_out,
                        )
                    )
                )
        return outputs


def eqhist_tf(im, nbins=256):
    """
    This function transform the value of the pixels of the image im to have an uniform distribution of intensities
    """
    imhist, bins = np.histogram(im.flatten(), nbins, normed=True)
    cdf = imhist.cumsum()
    cdf = 255 * cdf / cdf[-1]  # normalize

    # interpolate cdf to get thr new pixels values
    im2 = np.interp(im.flatten(), bins[:-1], cdf)

    return im2.reshape(im.shape)


def plot_max_dff_and_cellmap(
    cellsetfile, maxdff, eqhist=True, status_list=["accepted", "rejected", "undecided"]
):
    """
    This function plot the maxdff digure (with an optional transformation) add add the "borders" of the cell map
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

    """

    cell_set = isx.CellSet.read(cellsetfile)
    num_cells = cell_set.num_cells

    image_size = cell_set.get_cell_image_data(0).astype(np.float64)

    image = isx.Image.read(maxdff).get_data()
    if eqhist:
        image = eqhist_tf(image)

    plt.imshow(image, "Blues", interpolation="none", resample=False)
    for cell in range(num_cells):
        if cell_set.get_cell_status(cell) in status_list:
            cell_image = cell_set.get_cell_image_data(cell).astype(np.float64)
            plt.contour(
                cell_image > (0.8 * np.max(cell_image)),
                colors="red",
                linewidths=[2],
                alpha=0.3,
            )

    cell_set.flush()


def plot_max_dff_and_cellmap_fh(
    files_handler,
    idx=0,
    eqhist=True,
    cellsetname="cnmfe-cellset",
    status_list=["accepted", "rejected", "undecided"],
):
    """
    This function plot the maxdff figure (with an optional transformation) and adds the "borders" of the cell map

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

    """
    dataset = files_handler.get_results_filenames(cellsetname, op="MC", idx=[idx])[0]
    maxdff = files_handler.get_results_filenames("maxdff", op="MC", idx=[idx])[0]
    plot_max_dff_and_cellmap(dataset, maxdff, eqhist, status_list)


def interactive_reject_accept_cell(files_handler, cellset_names):
    """
    This function runs a simple GUI inside a jupyter notebook to see the calcium traces and accept/reject cells

    Parameters
    ----------
    files_handler : isx_files_handler
        isx_files_handler to use.
    cellset_names : str
        Name added to the results to describe the cellsets (usually related to the method use to detect cells).
    """
    import hvplot.pandas
    import panel as pn
    import pandas as pd

    pn.extension()

    def callback_cellinfile(target, event):
        cs = isx.CellSet.read(
            files_handler.get_results_filenames(cellset_names)[
                files_handler.recording_labels.index(event.new)
            ],
            read_only=True,
        )
        target.options = [cs.get_cell_name(x) for x in range(cs.num_cells)]

        target.value = target.options[0]
        target.param.trigger("value")

    select_files = pn.widgets.Select(
        name="Select Session", options=files_handler.recording_labels, width=130
    )
    select_cell = pn.widgets.Select(
        name="Select Cell", options=[], size=6, width=130
    )  # using group it possible to separete accepted and rejected
    accept_button = pn.widgets.Button(name="accept", button_type="success", width=60)
    reject_button = pn.widgets.Button(name="reject", button_type="danger", width=60)
    status = pn.widgets.StaticText(name="Status", value="", height=40, width=130)

    def change_cell_status(new_status):
        if status.value == new_status:
            return
        nfile = files_handler.recording_labels.index(select_files.value)
        cs = isx.CellSet.read(
            files_handler.get_results_filenames(cellset_names)[n],
            read_only=False,
        )
        n = select_cell.options.index(select_cell.value)
        cs.set_cell_status(n, new_status)
        cs.flush()
        status.value = cs.get_cell_status(n)

    accept_button.on_click(lambda x: change_cell_status("accepted"))
    reject_button.on_click(lambda x: change_cell_status("rejected"))

    @pn.depends(select_cell)
    def timeplot(cellvall):
        nfile = files_handler.recording_labels.index(select_files.value)
        n = select_cell.options.index(select_cell.value)

        cs = isx.CellSet.read(
            files_handler.get_results_filenames(cellset_names)[nfile],
            read_only=True,
        )
        status.value = cs.get_cell_status(n)
        time = np.linspace(
            0,
            cs.timing.num_samples * cs.timing.period.secs_float / 60,
            cs.timing.num_samples,
        )  # minutes
        data = pd.DataFrame(
            {"time (minutes)": time, "trace": cs.get_cell_trace_data(n)}
        )

        return data.hvplot.line(
            x="time (minutes)",
            y="trace",
            title=f"{select_files.value} {cellvall}",
            width=900,
        )

    select_files.link(
        select_cell, callbacks={"value": callback_cellinfile}, bidirectional=False
    )
    select_files.value = files_handler.recording_labels[0]
    select_files.param.trigger("value")

    return pn.Row(
        pn.Column(
            select_files, select_cell, status, pn.Row(accept_button, reject_button)
        ),
        timeplot,
    )
