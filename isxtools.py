import os
from pathlib import Path
from glob import glob
import numpy as np
import isx 
import matplotlib.pyplot as plt
import os
import log_file_fun
from datetime import datetime
import json

def get_efocus(main_video):
    gpio_set = isx.GpioSet.read(os.path.splitext(main_video)[0]+ '.gpio')
    efocus_values = gpio_set.get_channel_data(gpio_set.channel_dict['e-focus'])[1]
    efocus_values,efocus_counts = np.unique(efocus_values, return_counts=True)
    min_frames_per_efocus = 100
    video_efocus = efocus_values[efocus_counts>=min_frames_per_efocus]
    assert video_efocus.shape[0] < 4, 'Too many efocus detected, early frames issue.'
    return video_efocus.astype(int)

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
        Naming patterns for the files, and easy way to handle selection of one or multiple files from the same folder.
    outputsfolders : list
        Folder where the file structure (following only the data_subfolders) will be copied and the results written.
    processing_steps: list
        Naming steps will be use, adding one affter the previous ones.
    one_file_per_folder: bool
        If True it will check then one and only one file is found with the pattern in it's folder

    """

    def __init__(
        self,
        main_data_folder,
        data_subfolders,
        files_pattern,
        outputsfolders,
        processing_steps=["trim", "PP", "BP", "MC"],
        one_file_per_folder=True,
        recording_labels=None,
        parameters_path = os.path.join(os.path.dirname(__file__),'default_parameter.json') #change path
    ) -> None:
        if not isinstance(outputsfolders, list):
            outputsfolders = [outputsfolders]*len(data_subfolders)
        self.outputsfolders = [Path(o) for o in outputsfolders]
        self.data_subfolders = data_subfolders    
        for ofolder,sfolder in zip(self.outputsfolders, data_subfolders):
            os.makedirs(ofolder / sfolder, exist_ok=True)
        rec_names = []
        #check file existence and concatenate
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
        self._animals = None
        assert os.path.exists(parameters_path), 'parameters file does not exist'
        with open(parameters_path) as file:
            self.parameters_path = json.load(file) 
            
    def get_pair_filenames(self, operation):
        """
        This method returns the input/output pairs for the given operation step operation.
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

        for ofolder, fname, file in zip(self.outputsfolders,self.data_subfolders, self.rec_names):
            outputs.append(
                str(Path(ofolder, fname, Path(file).stem + suffix_out))
            )
            if suffix_in is not None:
                inputs.append(
                    str(Path(ofolder, fname, Path(file).stem + suffix_in))
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
        for ofolder,fname, file in zip(self.outputsfolders,self.data_subfolders, self.rec_names):
            outputs.append(
                str(Path(ofolder, fname, Path(file).stem + suffix_out))
            )
        return outputs

    def check_str_in_filenames(self, strlist,only_warning=True):
        """
        This method verify the order of the files. Checking that each element of the input list is included in the recording filenames
        Parameters
        ----------
        strlist : list
            List of strings to check
        only_warning: bool
            Default, True. If False, throw an exception
        """
        for x,y in zip(strlist,self.rec_names):
            if only_warning and x not in y:
                print(f"Warning: {x} not in {y}")
            else:
                assert x in y, f"Error {x} not in {y}"
    
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
                os.makedirs(os.path.join(
                            self.outputsfolders[i],
                            fname,
                            subfolder), exist_ok=True)
                outputs.append(
                    str(
                        Path(
                            self.outputsfolders[i],
                            fname,
                            subfolder,
                            Path(file).stem + suffix_out,
                        )
                    )
                )
        return outputs
    
    @property
    def animals(self):
        return self._animals

    @animals.setter
    def animals(self, animals):
        self._animals = np.array(animals)
        
    def remove_output_files(self, op):
        paths=self.get_filenames(op)
        for path in paths:
            if os.path.exists(path):
                os.remove(path)
                
    def run_step(self, op,overwrite=False, verbose=False, **kws):
        pairlist = self.get_pair_filenames(op)
        
        if overwrite:
            for input, output in pairlist:
                if os.path.exists(output):
                    os.remove(output)
                json_file = os.path.splitext(output)+'.json'
                if os.path.exists(json_file):
                    os.remove(json_file)
                    
        if op.startswith('BP'):
            print("Applying bandpass filter, please wait...\n")
            parameters = self.parameters_path['BP'].copy() 
            for key, value in kws.items():
                assert key in parameters, f'The parameter: {key} does not exist'
                parameters[key] = value
            spatial_filter_step(pairlist, parameters, verbose)
            return
        
        if op.startswith('PP'):
            print('Preprocessing, please wait...\n')
            parameters = self.parameters_path['PP'].copy() 
            for key, value in kws.items():
                assert key in parameters, f'The parameter: {key} does not exist'
                parameters[key] = value
            preprocess_step(pairlist, parameters, verbose)
            return
                        
        if op.startswith('MC'):
            print("Applying motion correction. Please wait...\n")
            parameters = self.parameters_path['MC'].copy() 
            for key, value in kws.items():
                assert key in parameters, f'The parameter: {key} does not exist'
                parameters[key] = value
            motion_correct_step(self,parameters, op, pairlist, verbose) 
            return
        if op.startswith('Trim'):
            print('Trim movies...\n')
            parameters = self.parameters_path['Trim'].copy() 
            for key, value in kws.items():
                assert key in parameters, f'The parameter: {key} does not exist'
                parameters[key] = value
            trim_movie(list, parameters , verbose)
            return
        if verbose:
            print("Error in operation name\n")                              
            

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


def plot_max_dff_and_cellmap_fh(
    files_handler,
    idx=0,
    eqhist=True,
    cellsetname="cnmfe-cellset",
    status_list=["accepted", "rejected", "undecided"],
    colors=['red','green', 'black'],
    op=None
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
    dataset = files_handler.get_results_filenames(cellsetname, op=op, idx=[idx])[0]
    maxdff = files_handler.get_results_filenames("maxdff", op=op, idx=[idx])[0]
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
            files_handler.get_results_filenames(cellset_names)[nfile],
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

def get_segment_from_movie(inputfile, outputfile, borders, keep_start_time=False,unit='minutes'):
    """
    This function gets a segment of a video to crete a new one. It's just an easy handler of isx.trim_movie

    Parameters
    ----------
    inputfile : str or Path
        Path of the input file to use
    outputfile : str or Path
        Path of the output file to use
    borders: list like 
        With two elements of the borders in minutes (integers) of the segment. Where negative times means from the end like indexing in a numpy array
    unit:
        the unit to use for border. It could be 'minutes' or 'seconds'
        
    Example:
    >>> get_segment_from_movie('inputfile.isxd','output.isxd',[-30, -1]) #last 30 minutes
    """
    assert len(borders)==2, 'borders must have two elements'
    assert unit in ['minutes' , 'seconds'], '''unit could be 'minutes' or 'seconds'. '''
    movie = isx.Movie.read(inputfile)
    if unit=='minutes':
         unitfactor = 60
    else:
         unitfactor = 1
    duration_unit = movie.timing.num_samples  * movie.timing.period.to_msecs() / (1000 * unitfactor) 
    timing_period_unit = movie.timing.period.to_msecs() / (1000 * unitfactor)
    numframes = movie.timing.num_samples
    
    movie.flush()

    crop_segments = []
    assert borders[0] > -duration_unit, 'asking for a segment from {minutes[0]} {unit} before the end, but video is {duration_minutes} {unit} long.'
    assert borders[0] < duration_unit, 'asking for a segment from {minutes[0]} {unit}, but video is {duration_minutes} {unit} long.'
    assert borders[1] < duration_unit, 'asking for a segment up to {minutes[0]} {unit}, but video is {duration_minutes} {unit} long.'
    assert borders[1] > -duration_unit, 'asking for a segment up to {minutes[0]} {unit} before the end, but video is {duration_minutes} {unit} long.'

    #remove fist frames:
    if borders[0]!=0 and borders[0]!=-duration_unit: #don't cut if the segments are from the beggining or just exactlt duration before the end
        if borders[0]<0:
            end1 =  (duration_unit + borders[0])/timing_period_unit -1
        else:
            end1 =  borders[0] / timing_period_unit - 1
        crop_segments.append([0, int(end1)])
    #remove last frames:
    if borders[1]!=-1 and  borders[1]!=duration_unit:  #don't cut if the segments are up to the end or just exactlt duration
        if borders[1]<0:
           start1 =  (duration_unit + borders[1] + 1) / timing_period_unit + 1
        else:
           start1 =  borders[1] / timing_period_unit + 1
        crop_segments.append([int(start1), numframes])

    if os.path.exists(outputfile):
        os.remove(outputfile)
    if len(crop_segments)==0:
        print('no trim need it')
        return
    isx.trim_movie(
        input_movie_file=inputfile,  
        output_movie_file=outputfile, 
        crop_segments=np.array(crop_segments),
        keep_start_time=keep_start_time
    )
    """
    This function call isx.motion_correct funtion

    Parameters
    ----------
    filehandler 
        To get information from files
    list : input, output files from de op "MC"
        For optimization.
    verbose: bool
        if is True, print some information
        
    """
def motion_correct_step(filehandler, parameters, op, pairlist, verbose = False):
    actual_idx = filehandler.processing_steps.index(op)
    if actual_idx != 0:
        op_prev = filehandler.processing_steps[actual_idx - 1]
        mean_proj_files = filehandler.get_results_filenames("mean_image", op=op_prev)
        translation_files = filehandler.get_results_filenames("translations.csv", op=op_prev)
        crop_rect_files = filehandler.get_results_filenames("crop_rect.csv", op=op_prev)
    else:
        assert "Not Implemented"
    for i, (input, output) in enumerate(pairlist):
        if consistency_output_json(parameters, output):
            if verbose:
                print("Already ran with this parameters")
            continue
        new_data = {'input_movie_files': input, 'output_movie_files': output,
                    'reference_file_name': mean_proj_files[i],'output_translation_files': translation_files[i],
                    'output_crop_rect_file': crop_rect_files[i]
                    }
        parameters.update(new_data)
        isx.motion_correct( 
            **parameters  
        )
        parameters['function'] = 'motion_correct'
        parameters['isx_version'] = isx.__version__
        write_log_file(parameters)
        if verbose:
            print("{} motion correction completed".format(output))  
                
def preprocess_step(list, parameters, verbose = False):
    for input, output in list:
        if isinstance(parameters['sp_downsampling'],str):
            res_idx, value = parameters['sp_downsampling'].split('_')
            if res_idx == 'maxHeight':
                idx_resolution = 0
            elif res_idx == 'maxWidth':
                idx_resolution = 1
            else:
               assert False, 'error in sp_downsampling parameter value' 
            movie = isx.Movie.read(input) 
            resolution=movie.spacing.num_pixels
            parameters['sp_downsampling'] = np.ceil(resolution[idx_resolution] / float(value))
                
        parameters.update({'input_movie_files': input, 'output_movie_files': output})
        if consistency_output_json(parameters, output):
            if verbose:
                print("Already ran with this parameters")
            continue
        isx.preprocess(
            **parameters
        )
        parameters['function'] = 'preprocess'
        parameters['isx_version'] = isx.__version__
        write_log_file(parameters)
        if verbose:
            print("{} preprocessing completed".format(output))  
                
def spatial_filter_step(list, parameters, verbose = False):
    for input, output in list:      
        parameters.update({'input_movie_files': input, 'output_movie_files': output})
        if consistency_output_json(parameters, output):
            if verbose:
                print("Already ran with this parameters")
            continue
        isx.spatial_filter(
        **parameters
        )
        parameters['function'] = 'spatial_filter'
        parameters['isx_version'] = isx.__version__
        write_log_file(parameters)
        if verbose:
            print("{} bandpass filtering completed".format(output)) 
                
def trim_movie(list, user_parameters, verbose = False):
    assert user_parameters['video_len'] is not None, 'Trim movie requires parameter video len'
    for input, output in list:
        parameters = {'input_movie_files': input, 'output_movie_files': output}
        movie = isx.Movie.read(input) 
        sr = 1/(movie.timing.period.to_msecs()/1000)
        endframe =  user_parameters['video_len']*sr
        maxfileframe = movie.timing.num_samples()+1
        if maxfileframe < endframe:
            parameters['maxfileframe'] = maxfileframe
            parameters['endframe'] = endframe
            assert("max time selected is greater than duration of the video")
        parameters['crop_segments'] = np.array([[endframe,maxfileframe]])
        if consistency_output_json(parameters, output):
            if verbose:
                print("Already ran with this parameters")
            continue
        isx.trim_movie(
            **parameters
        )
        if verbose:
            print('{} trimming completed'.format(output))
        assert os.path.exists(output),  'File not created: {}'.format(output)
        parameters['function'] = 'spatial_filter'
        parameters['isx_version'] = isx.__version__
        write_log_file(parameters)
        
def write_log_file(data):
    log_path = os.path.splitext(data['output_movie_files'])[0] + ".json"
    actual_date = datetime.utcnow()
    data['input_modification_data'] = None
    input_json = os.path.splitext(data['input_movie_files'])[0] + ".json"
    if os.path.exists(input_json):
        with open(input_json) as file:
            input_data = json.load(file)
        data['input_modification_data']=input_data['date']
    data['date']= actual_date.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, 'w') as file:
        json.dump(data, file, indent = 4)
        
def is_same_parameters(json_file, new_data):
    with open(json_file) as file:
        prev_data = json.load(file)
    for key, value in new_data.items():
        if prev_data[key] != value:
            return False
    return True

#Return True if the input file was not modified or if it does not exist
def is_same_input(input, output):
    if os.path.exists(input):
        with open(output) as out_file:
            out_data = json.load(out_file)
        with open(input) as in_file:
            in_data = json.load(in_file)
        if out_data['input_modification_data'] != in_data['date']:
            return False
    return True


def consistency_output_json(parameters, output):
    json_file = os.path.splitext(output)+'.json'
    if os.path.exists(json_file):
        if os.path.exists(output):
            input_json = os.path.splitext(input)+'.json'
            if is_same_input(input_json, json_file):
                if is_same_parameters(json_file, parameters):
                    return True
        os.remove(json_file)
    if os.path.exists(output):
        os.remove(output)
    return False