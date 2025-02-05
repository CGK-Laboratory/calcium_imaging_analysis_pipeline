import os
from pathlib import Path
from glob import glob
import numpy as np
import isx
import matplotlib.pyplot as plt
import os
from datetime import datetime
import hvplot.pandas
import panel as pn
import pandas as pd


def interactive_reject_accept_cell(RecordingHandler, cellset_names: str) -> pn.Row:
    """
    This function runs a simple GUI inside a jupyter notebook to see the calcium traces
    and accept/reject cells

    Parameters
    ----------
    RecordingHandler : isx_RecordingHandler
        isx_RecordingHandler to use.
    cellset_names : str
        Name added to the results to describe the cellsets (usually related to the method use to detect cells).

    Returns
    -------
    np.Row
        panel layout
    """
    pn.extension()
    file_list = RecordingHandler.get_results_filenames(cellset_names, single_plane=True)

    def callback_cellinfile(target, event):
        cs = isx.CellSet.read(
            file_list[RecordingHandler.p_recording_labels.index(event.new)],
            read_only=True,
        )
        target.options = [cs.get_cell_name(x) for x in range(cs.num_cells)]

        target.value = target.options[0]
        target.param.trigger("value")

    select_files = pn.widgets.Select(
        name="Select Session", options=RecordingHandler.recording_labels, width=130
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
        nfile = RecordingHandler.recording_labels.index(select_files.value)
        cs = isx.CellSet.read(file_list[nfile], read_only=False)
        n = select_cell.options.index(select_cell.value)
        cs.set_cell_status(n, new_status)
        cs.flush()
        status.value = cs.get_cell_status(n)

    accept_button.on_click(lambda x: change_cell_status("accepted"))
    reject_button.on_click(lambda x: change_cell_status("rejected"))

    @pn.depends(select_cell)
    def timeplot(cellvall):
        nfile = RecordingHandler.recording_labels.index(select_files.value)
        n = select_cell.options.index(select_cell.value)

        cs = isx.CellSet.read(
            file_list[nfile],
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
    select_files.value = RecordingHandler.p_recording_labels[0]
    select_files.param.trigger("value")

    return pn.Row(
        pn.Column(
            select_files, select_cell, status, pn.Row(accept_button, reject_button)
        ),
        timeplot,
    )
