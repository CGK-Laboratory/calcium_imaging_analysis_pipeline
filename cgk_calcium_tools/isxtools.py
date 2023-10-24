import os
from pathlib import Path
from glob import glob
import numpy as np
import isx 
from datetime import datetime
import json
from typing import Union, Tuple

def ifstr2list(x)->list:
    if isinstance(x, list):
        return x
    return [x]

class isx_files_handler:
    """
    This class helps to iterate over files for inscopix processing

    Parameters
    ----------
    main_data_folder : str, path or list
        Folder with data, after this path the output will share the folder structure. 
        If it's a list each element should correspond with each data_subfolders 
        and files_pattern.
    data_subfolders : list
        Where each file (or files) following each files_pattern element are.
    files_pattern : list
        Naming patterns for the files, and easy way to handle selection of one or 
        multiple files from the same folder.
    outputsfolders : list
        Folder where the file structure (following only the data_subfolders) will be 
        copied and the results written.
    processing_steps: list
        Naming steps will be use, adding one affter the previous ones.
    one_file_per_folder: bool
        If True it will check then one and only one file is found with the pattern in 
        its folder

    """

    def __init__(
        self,
        main_data_folder:Union[str, list]='.',
        outputsfolders:Union[str, list]='.',
        data_subfolders:Union[str, list]='.',
        files_pattern:Union[str, list]='.isx',
        processing_steps:list=["PP", "TR", "BP", "MC"],
        one_file_per_folder:bool=True,
        recording_labels:Union[str, None]=None,
        parameters_path:str=
            os.path.join(os.path.dirname(__file__),'default_parameter.json')
    )->None: 

        
        lists_inputs = {
            'main_data_folder': ifstr2list(main_data_folder),
            'outputsfolders': ifstr2list(outputsfolders),
            'data_subfolders': ifstr2list(data_subfolders),
            'files_pattern': ifstr2list(files_pattern)
        }
        len_list_variables = np.unique([len(v) for v in lists_inputs.values()])
        len_list_variables = len_list_variables[len_list_variables>1]

        if len(len_list_variables)>0:
            assert len(np.unique(len_list_variables))==1, \
                "the list inputs should have the same length"
            len_list = len_list_variables[0]
        else:
            len_list = 1

        for k,v in lists_inputs.items():
            if len(v)!=len_list:
                lists_inputs[k]=v*len_list #this will extend the single inputs 

        self.rec_subfolders = []
        self.outputsfolders  = []
        self.rec_paths = []
        for mainf,f,fpatter,outf in zip(lists_inputs['main_data_folder'],
            lists_inputs['data_subfolders'],lists_inputs['files_pattern'],
            lists_inputs['outputsfolders']):

            files = glob(str(Path(mainf)/f/fpatter), recursive=False)
            
            if one_file_per_folder:
                assert len(files) == 1, "None or multiple files found for {}.".format(
                    str(Path(mainf) / f / fpatter)
                )
                self.rec_paths.append(files[0])
                self.outputsfolders.append(Path(outf)/f)
            else:
                assert len(files) > 0, "No file found for {}.".format(
                    str(Path(mainf) / f / fpatter)
                )
                rec2add = [r for r in files if f not in self.rec_paths]
                self.rec_paths.extend(rec2add)
                self.outputsfolders.extend([Path(outf)/f for r in rec2add])

        for ofolder in self.outputsfolders:
            os.makedirs(ofolder, exist_ok=True)

        self.processing_steps = processing_steps
        if recording_labels is None:
            self.recording_labels = self.rec_paths
        else:
            assert len(recording_labels)==len(self.rec_paths), \
                'Recordings and reconding labels should have same length'
            self.recording_labels = recording_labels

        assert os.path.exists(parameters_path), 'parameters file does not exist'
        with open(parameters_path) as file:
            self.default_parameters= json.load(file) 

        #Lookig for multiplanes:
        self.p_rec_paths = []
        self.p_outputsfolders = []
        self.p_recording_labels = []
        self.focus_files = {}
        self.efocus = []
        for i, rec in enumerate(self.rec_paths):
            gpio_file = os.path.splitext(rec)[0]+ '_gpio.isxd'
            if os.path.exists(gpio_file):
                efocus = get_efocus(gpio_file)
            else:
                video= isx.Movie.read(rec)
                get_acquisition_info= video.get_acquisition_info().copy()
                del video #usefull for windows
                if 'Microscope Focus' in get_acquisition_info:
                    assert isx.verify_deinterleave(rec,
                        get_acquisition_info['Microscope Focus']) ,\
                            f'Info {rec}: Multiple Microscope Focus but not gpio file'
                    efocus = [get_acquisition_info['Microscope Focus']]
                else:
                    efocus = [0]
                    print(f'Info: Unable to verify Microscope Focus config in: {rec}')
            if len(efocus)==1:
                self.focus_files[rec]=[rec]
                self.p_rec_paths.append(rec)
                self.p_recording_labels.append(self.recording_labels[i])
                self.efocus.append(efocus)
            else:
                self.efocus.append(efocus)
                pr = Path(rec)
                efocus_filenames = [pr.stem + '_'+ str(ef) + pr.suffix for ef in efocus]
                self.focus_files[rec]=[str(self.outputsfolders[i]/ef)  
                                            for ef in efocus_filenames]
                self.p_recording_labels.extend([self.recording_labels[i]+f'_{ef}'
                                                    for ef in efocus])
                self.p_rec_paths.extend(self.focus_files[rec])
            self.p_outputsfolders.extend([self.outputsfolders[i]]*len(efocus))

    def de_interleave(self, overwrite:bool=False)->None:
        print('de_interleaving movies, please wait...')
        for (main_file,planes_fs), focus in zip(self.focus_files.items(), self.efocus):
            if len(focus)>1: #has multiplane
                sp_file2do = []
                planes2do = []
                for sp_file, single_plane in zip(planes_fs, focus):
                    if os.path.exists(sp_file):
                        if overwrite:
                            os.remove(sp_file)
                        else:
                            continue
                    sp_file2do.append(sp_file)
                    planes2do.append(single_plane)
                if len(sp_file2do)>0: #has files to run
                    isx.de_interleave(main_file, sp_file2do, planes2do)
        print('done')

    def get_pair_filenames(self, operation:str)->Tuple[list,list]:
        """
        This method returns the input/output pairs for the given operation step.
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
            inputs = self.p_rec_paths
        else:
            suffix_in = "-" + "-".join(self.processing_steps[:opi]) + ".isxd"

        for ofolder, file in zip(self.p_outputsfolders, self.p_rec_paths):
            outputs.append(
                str(Path(ofolder, Path(file).stem + suffix_out))
            )
            if suffix_in is not None:
                inputs.append(
                    str(Path(ofolder, Path(file).stem + suffix_in))
                )

        return zip(inputs, outputs)
    
    def get_filenames(self, op:Union[None, str]=None)->list:
        """
        This method return the filenames of the files create after the operation op.
        Parameters
        ----------
        op : str or None
            Operation in the steps. If None, it will be the final one. Default: None
        """
        if op is None:
            op = self.processing_steps[-1]
        assert (op in self.processing_steps
        ), f"operation must be in the list {self.processing_steps}, got: {op}"

        opi = self.processing_steps.index(op)
        outputs = []
        suffix_out = "-" + "-".join(self.processing_steps[: opi + 1]) + ".isxd"
        
        for ofolder, file in zip(self.p_outputsfolders, self.p_rec_paths):
            outputs.append(
                str(Path(ofolder, Path(file).stem + suffix_out))
            )
        return outputs

    def check_str_in_filenames(self, strlist:list,only_warning:bool=True)->None:
        """
        This method verify the order of the files. Checking that each element of the 
        input list is included in the recording filenames
        Parameters
        ----------
        strlist : list
            List of strings to check
        only_warning: bool
            Default, True. If False, throw an exception
        """
        for x,y in zip(strlist,self.rec_paths):
            if only_warning and x not in y:
                print(f"Warning: {x} not in {y}")
            else:
                assert x in y, f"Error {x} not in {y}"
    
    def get_results_filenames(self, name:str, op:Union[None,str]=None, subfolder:str="",
        idx:Union[None,int]=None, proccesing:bool=True)->list:

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
        
        if proccesing:
            file_list = self.p_rec_paths
            ofolders = self.p_outputsfolders
        else:
            file_list  = self.rec_paths
            ofolders = self.outputsfolders
        outputs = []
        for i,(file,ofolder) in enumerate(zip(file_list, ofolders)):
            if idx is None or i in idx:
                os.makedirs(os.path.join(ofolder,
                            subfolder), exist_ok=True)
                outputs.append(
                    str(
                        Path(ofolder,subfolder,
                            Path(file).stem + suffix_out,
                        )
                    )
                )
        return outputs
    
    def remove_output_files(self, op):
        paths=self.get_filenames(op)
        for path in paths:
            if os.path.exists(path):
                os.remove(path)
            json_file = json_filename(path)
            if os.path.exists(json_file):
                os.remove(json_file) 
   
    def run_step(self, op:str,overwrite:bool=False, verbose=False, **kws):
        """This functions runs

        Parameters
        ----------
        op : str
            Preprocessing operation to run
        overwrite : bool, optional
            Remove results and recompute them, by default False
        verbose : bool, optional
            Show additional messages, by default False
        """
        pairlist = self.get_pair_filenames(op)
        
        if overwrite:
            for _, output in pairlist:
                if os.path.exists(output):
                    os.remove(output)
                json_file = json_filename(output)
                if os.path.exists(json_file):
                    os.remove(json_file)
        
        
        steps_list = {'PP':'preprocessing','BP':'spatial_filter',
            'MC':'motion_correct','TR':'trim'}

        operation = [x for k,x in steps_list.items() if op.startswith(k)]
        assert len(operation)==1, f'Step {op} not starts with {steps_list}.'
        operation = operation[0]
        
        parameters = self.default_parameters[operation].copy()
        for key, value in kws.items():
            assert key in parameters, f'The parameter: {key} does not exist'
            parameters[key] = value

        if self.processing_steps.index(op)==0: #if first step
            #check if all exist
            for finput, _ in self.get_pair_filenames(op):
                assert os.path.exists(finput),f'''File {finput} not exist: 
                    Run .de_interleave() to De-interleave multiplane movies'''

        if op.startswith('PP'):
            print('Preprocessing movies, please wait...')
            preprocess_step(pairlist, parameters, verbose)
        
        if op.startswith('BP'):
            print("Applying bandpass filter, please wait...")
            spatial_filter_step(pairlist, parameters, verbose)
                        
        if op.startswith('MC'):
            print("Applying motion correction. Please wait...")
            actual_idx = self.processing_steps.index(op)
            if actual_idx != 0:
                op_prev = self.processing_steps[actual_idx - 1]
                translation_files = self.get_results_filenames("translations.csv",
                                                                op=op_prev)
                crop_rect_files = self.get_results_filenames("crop_rect.csv",
                                                                op=op_prev)
            else:
                assert "Not Implemented"

            motion_correct_step(translation_files,crop_rect_files,parameters,
                                pairlist,verbose) 
        if op.startswith('TR'):
            print('Trim movies...')
            trim_movie(pairlist, parameters , verbose)
        
        if verbose:
            assert False, "Error in operation name"                           
        print ('done')

    def project_movie(self, input_name:str='dff',output_name:str='maxdff',
                    operation=None,overwrite=False, verbose=False, **kws):
        if overwrite:
            for output in self.get_results_filenames(output_name, op=operation):
                if os.path.exists(output):
                    os.remove(output)
                json_file = json_filename(output)
                if os.path.exists(json_file):
                    os.remove(json_file)
        parameters = self.default_parameters['project_movie'].copy() 
        
        for key, value in kws.items():
            assert key in parameters, f'The parameter: {key} does not exist'
            parameters[key] = value

        for input, output in zip(self.get_results_filenames(input_name, op=operation), 
                                self.get_results_filenames(output_name, op=operation)):
            new_data = {'input_movie_files': input, 'output_image_file': output}
            parameters.update(new_data)
            if same_json_or_remove(parameters, input_files_keys=['input_movie_files'],
                output=output, verbose=verbose):
                continue
            isx.project_movie(**del_key(parameters,'comments'))
            write_log_file(parameters,{'function':'project_movie'},
                input_files_keys=['input_movie_files'], 
                output_file_key='output_image_file')





    def create_dff(self, overwrite=False, verbose=False, **kws):
        if overwrite:
            for output in self.get_results_filenames("dff", op=None):
                if os.path.exists(output):
                    os.remove(output)
                json_file = json_filename(output)
                if os.path.exists(json_file):
                    os.remove(json_file)
        parameters = self.default_parameters['dff'].copy() 
        
        for key, value in kws.items():
            assert key in parameters, f'The parameter: {key} does not exist'
            parameters[key] = value

        for input, output in zip(self.get_filenames(op=None), 
                                    self.get_results_filenames("dff", op=None)):
            new_data = {'input_movie_files': input, 'output_movie_files': output}
            parameters.update(new_data)
            if same_json_or_remove(parameters, input_files_keys=['input_movie_files'],
                output=output, verbose=verbose):
                continue
            isx.dff(**del_key(parameters,'comments'))
            write_log_file(parameters,{'function':'dff'},
                input_files_keys=['input_movie_files'], 
                output_file_key='output_movie_files')

    def extract_cells(self, alg:str, overwrite:bool=False, verbose:bool=False,
        cells_params:Union[dict,None]=None, detection_params:Union[dict,None]=None,
        accept_reject_params:Union[dict,None]=None,
         multiplane_params:Union[dict,None]=None)->None:
        """This function run a cell extraction algorithm, detect events, auto accept_reject and 
        multiplane registration

        Parameters
        ----------
        alg : str
            Cell extraction algorithm: 'cnmfe' or 'pca-ica'
        overwrite : bool, optional
            Force compute everything again, by default False
        verbose : bool, optional
            Show additional messages, by default False
        cells_params : Union[dict,None], optional
            Parameters for cell extraction, by default None
        detection_params : Union[dict,None], optional
            Parameters for event detection, by default None
        accept_reject_params : Union[dict,None], optional
            Parameters for automatic accept_reject cell, by default None
        multiplane_params : Union[dict,None], optional
            Parameters for multiplane registration, by default None
        """
        assert alg in ['pca-ica','cnmfe'],"alg must be 'pca-ica' or 'cnmfe'."
        
        #extract cells
        parameters = self.default_parameters[alg].copy() 
        if cells_params is not None:
            for key, value in cells_params.items():
                assert key in parameters, f'The parameter: {key} does not exist'
                parameters[key] = value

        if alg=='pca-ica':
            cell_det_fn = isx.pca_ica
            inputs_files = self.get_results_filenames("dff", op="MC") #pca uses dff:
        elif alg=='cnmfe':
            cell_det_fn = isx.run_cnmfe
            inputs_files = self.get_filenames(op=None)
        else:
            raise "alg must be 'pca-ica' or 'cnmfe'."
        cellsets = self.get_results_filenames(f"{alg}", op=None)

        if overwrite:
            for fout in cellsets:
                if os.path.exists(fout):
                    os.remove(fout)
                    json_file = json_filename(fout)
                    if os.path.exists(json_file):
                        os.remove(json_file)


        for input, output in zip(inputs_files, cellsets):
            new_data = {'input_movie_files': input, 'output_cell_set_files': output}
            parameters.update(new_data)
            if same_json_or_remove(parameters,input_files_keys=['input_movie_files'],
                output=output, verbose=verbose):
                continue
            cell_det_fn(**del_key(parameters,'comments'))

            if not os.path.exists(output):
                print(f"Warning: Algorithm {alg}, failed to create file: {output}.\n" +
                                "Empty cellmap created with its place")
                movie = isx.Movie.read(input)
                cell_set = isx.CellSet.write(output, movie.timing, movie.spacing)
                image_null = np.zeros(cell_set.spacing.num_pixels).astype(np.float32)
                trace_null = np.zeros(cell_set.timing.num_samples).astype(np.float32)
                cell_set.set_cell_data(0, image_null, trace_null, '')
                cell_set.flush()
                del cell_set
                del movie

            write_log_file(parameters, {'function':alg},
                input_files_keys = ['input_movie_files'], 
                output_file_key = 'output_cell_set_files')
        if verbose:
            print('Cell extraccion, done')
        #detect events
        ed_parameters = self.default_parameters['event_detection'].copy() 
        if detection_params is not None:
            for key, value in detection_params.items():
                assert key in ed_parameters, f'The parameter: {key} does not exist'
                ed_parameters[key] = value

        for input, output in zip(cellsets,
            self.get_results_filenames(f"{alg}-ED", op=None)):
            
            new_data = {'input_cell_set_files': input, 'output_event_set_files': output}
            ed_parameters.update(new_data) 
            if same_json_or_remove(ed_parameters, output=output, verbose=verbose,
                input_files_keys=['input_cell_set_files']):
                continue
            try:
                isx.event_detection(**del_key(ed_parameters,'comments'))
            except Exception as e:
                print(f"Warning: Event_detection, failed to create file: {output}.\n" +
                                "Empty file created with its place")
                cell_set = isx.CellSet.read(input)
                evset = isx.EventSet.write(output, cell_set.timing, [''])
                evset.flush()
                del evset
                del cell_set
            write_log_file(ed_parameters,{'function':'event_detection'},
                        input_files_keys = ['input_cell_set_files'],
                        output_file_key ='output_event_set_files')

        if verbose:
            print('Event detection, done')

        #accept reject cells
        ar_parameters = self.default_parameters['accept_reject'].copy() 
        if accept_reject_params is not None:
            for key, value in accept_reject_params.items():
                assert key in ar_parameters, f'The parameter: {key} does not exist'
                ar_parameters[key] = value

        for input_cs, input_ev, config_json in zip(cellsets,
                self.get_results_filenames(f"{alg}-ED", op=None),
                self.get_results_filenames(f"{alg}-accept_reject", op=None)):

            new_data = {'input_cell_set_files':input_cs,
                        'input_event_set_files':input_ev}
            ar_parameters.update(new_data) 
            try:
                isx.auto_accept_reject(**del_key(ar_parameters,'comments'))
            except Exception as e:
                if verbose:
                    print(e)
            write_log_file(ar_parameters,{'function':'accept_reject','config_json':config_json},
                input_files_keys=['input_cell_set_files','input_event_set_files'],
                output_file_key='config_json')
        if verbose:
            print('accept reject cells, done')

        if len([True for x in self.focus_files.values() if len(x)>1])==0:
            return
        #multiplane_registration
        if verbose:
            print('Starting multiplane registration:...')
        parameters = self.default_parameters['multiplane_registration'].copy() 
        if multiplane_params is not None:
            for key, value in multiplane_params.items():
                assert key in parameters, f'The parameter: {key} does not exist'
                parameters[key] = value
        
        for main_file,single_planes in self.focus_files.items():
            if len(single_planes)==1: #doesn't have multiplane
                continue
            input_cell_set_files = self.get_results_filenames(f"{alg}", op=None, 
                            idx=[self.p_rec_paths.index(f) for f in single_planes])
            idx = [self.rec_paths.index(main_file)]
            output_cell_set_file = self.get_results_filenames(f"{alg}", op=None,
                            idx=idx, proccesing=False)[0]
            ed_file = self.get_results_filenames(f"{alg}-ED", op=None, 
                            idx=idx, proccesing=False)[0]
            ar_cell_set_file = self.get_results_filenames(f"{alg}-accept_reject", 
                        op=None, idx=idx, proccesing=False)[0]

            
            new_data = {'input_cell_set_files': input_cell_set_files,
                        'output_cell_set_file': output_cell_set_file}
            mpr_parameters=self.default_parameters['multiplane_registration'].copy()
            mpr_parameters.update(new_data) 

            if same_json_or_remove(mpr_parameters, output=output_cell_set_file,
                verbose=verbose, input_files_keys=['input_cell_set_files']):
                continue

            try:
                isx.multiplane_registration(**del_key(mpr_parameters,'comments'))
            except Exception as e:
                # Code to handle the exception
                print(f"Exception: {e}")

                if not os.path.exists(output_cell_set_file):
                    print(f"Warning: File: {output_cell_set_file} not generated.\n" +
                                    "Empty cellmap created in its place")
                    cell_set_plane = isx.CellSet.read(input_cell_set_files[0])
                    cell_set = isx.CellSet.write(output_cell_set_file, cell_set_plane.timing, cell_set_plane.spacing)
                    image_null = np.zeros(cell_set.spacing.num_pixels,dtype=np.float32)
                    trace_null = np.zeros(cell_set.timing.num_samples,dtype=np.float32)
                    cell_set.set_cell_data(0, image_null, trace_null, '')
                    cell_set.flush()
                    del cell_set
                    del cell_set_plane

            write_log_file(mpr_parameters,{'function': 'multiplane_registration'},
                                input_files_keys = 'input_cell_set_files', 
                                output_file_key ='output_cell_set_file')
            new_data = {'input_cell_set_files': output_cell_set_file, 
                        'output_event_set_files': ed_file}
            ed_parameters.update(new_data) 
            if same_json_or_remove(ed_parameters, output=ed_file, verbose=verbose,
                input_files_keys=['input_cell_set_files']):
                continue
            try:
                isx.event_detection(**del_key(ed_parameters,'comments'))
            except Exception as e:
                print(f"Warning: Event_detection, failed to create file: {ed_file}.\n" +
                                "Empty file created with its place")
                cell_set = isx.CellSet.read(output_cell_set_file)
                evset = isx.EventSet.write(ed_file, cell_set.timing, [''])
                evset.flush()
                del evset
                del cell_set
            
            write_log_file(ed_parameters,{'function':'event_detection'},
                input_files_keys = ['input_cell_set_files'], 
                output_file_key ='output_event_set_files')

            new_data = {'input_cell_set_files': output_cell_set_file,
                    'input_event_set_files': ed_file}
            ar_parameters.update(new_data) 
            try:
                isx.auto_accept_reject(**del_key(ar_parameters,'comments'))
            except Exception as e:
                if verbose:
                    print(e)
            write_log_file(ar_parameters,{'function':'accept_reject','config_json':ar_cell_set_file},
                input_files_keys = ['input_cell_set_files','input_event_set_files'],
                output_file_key ='config_json')
        print('done')
                



def get_segment_from_movie(inputfile, outputfile, borders, 
                            keep_start_time=False,unit='minutes'):
    """
    This function gets a segment of a video to create a new one. It's just an easy 
    handler of isx.trim_movie

    Parameters
    ----------
    inputfile : str or Path
        Path of the input file to use
    outputfile : str or Path
        Path of the output file to use
    borders: list like 
        With two elements of the borders in minutes (integers) of the segment. 
        Where negative times means from the end like indexing in a numpy array
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
    numframes = movie.timing.num_samples     
    duration_unit = numframes * movie.timing.period.to_msecs() / (1000 * unitfactor) 
    timing_period_unit = movie.timing.period.to_msecs() / (1000 * unitfactor)
    
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
    # and don't cut if the segments are up to the end or just the exact duration
    if borders[1]!=-1 and  borders[1]!=duration_unit:  
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

def motion_correct_step(translation_files,crop_rect_files, 
                        parameters, op, pairlist, verbose = False)->None:
    for i, (input, output) in enumerate(pairlist):
        if same_json_or_remove(parameters, input_files_keys=['input_movie_files'],
            output=output, verbose=verbose):
            continue
        new_data = {'input_movie_files': input, 'output_movie_files': output,
                    'output_translation_files': translation_files[i],
                    'output_crop_rect_file': crop_rect_files[i]
                    }
        parameters.update(new_data)
        isx.motion_correct(**del_key(parameters,'comments'))
        write_log_file(parameters, {'function':'motion_correct'},
            input_files_keys = ['input_movie_files'], 
            output_file_key = 'output_movie_files')
        if verbose:
            print("{} motion correction completed".format(output))
                
def preprocess_step(pairlist, parameters, verbose = False):
    for input, output in pairlist:
        if isinstance(parameters['spatial_downsample_factor'],str):
            res_idx, value = parameters['spatial_downsample_factor'].split('_')
            if res_idx == 'maxHeight':
                idx_resolution = 0
            elif res_idx == 'maxWidth':
                idx_resolution = 1
            else:
               assert False, 'error in sp_downsampling parameter value' 
            movie = isx.Movie.read(input) 
            resolution=movie.spacing.num_pixels
            del movie
            parameters['spatial_downsample_factor'] = np.ceil(resolution[idx_resolution] / float(value))
                
        parameters.update({'input_movie_files': input, 'output_movie_files': output})
        if same_json_or_remove(parameters, input_files_keys=['input_movie_files'],
            output=output, verbose=verbose):
            continue
        isx.preprocess(**del_key(parameters,'comments'))
        write_log_file(parameters,{'function':'preprocess'},
            input_files_keys = ['input_movie_files'], output_file_key = 'output_movie_files')
        if verbose:
            print("{} preprocessing completed".format(output))  
                
def spatial_filter_step(list, parameters, verbose = False):
    for input, output in list:      
        parameters.update({'input_movie_files': input, 'output_movie_files': output})
        if same_json_or_remove(parameters, input_files_keys = ['input_movie_files'],
            output=output, verbose=verbose):
            continue
        isx.spatial_filter(**del_key(parameters,'comments'))
        write_log_file(parameters, {'function':'spatial_filter'},
            input_files_keys = ['input_movie_files'], output_file_key = 'output_movie_files')
        if verbose:
            print("{} bandpass filtering completed".format(output)) 
                
def trim_movie(pairlist, user_parameters, verbose = False):
    assert user_parameters['video_len'] is not None, 'Trim movie requires parameter video len'
    for input, output in pairlist:
        parameters = {'input_movie_file': input, 'output_movie_file': output}
        movie = isx.Movie.read(input) 
        sr = 1/(movie.timing.period.to_msecs()/1000)
        endframe =  user_parameters['video_len']*sr
        maxfileframe = movie.timing.num_samples+1
        assert maxfileframe >= endframe, "max time > duration of the video"
        parameters['crop_segments'] = [[endframe,maxfileframe]]
        if same_json_or_remove(parameters, input_files_keys = ['input_movie_file'],
            output=output, verbose=verbose):
            continue
        isx.trim_movie(**del_key(parameters,'comments'))
        if verbose:
            print('{} trimming completed'.format(output))
        write_log_file(parameters, {'function':'trimming'},
            input_files_keys = ['input_movie_file'],
            output_file_key = 'output_movie_file')

def get_efocus(gpio_file):
    gpio_set = isx.GpioSet.read(gpio_file)
    efocus_values = gpio_set.get_channel_data(gpio_set.channel_dict['e-focus'])[1]
    efocus_values,efocus_counts = np.unique(efocus_values, return_counts=True)
    min_frames_per_efocus = 100
    video_efocus = efocus_values[efocus_counts>=min_frames_per_efocus]
    assert video_efocus.shape[0] < 4, f'{gpio_file}: Too many efocus detected, early frames issue.'
    return list(video_efocus.astype(int))

def write_log_file(params, extra_params={}, input_files_keys = ['input_movie_files'],
    output_file_key = 'output_movie_files'):
    data = {}
    data.update(params)
    data.update(extra_params)
    data['isx_version'] = isx.__version__

    log_path = json_filename(data[output_file_key])
    actual_date = datetime.utcnow()
    data['input_modification_date'] = None
    if not isinstance(input_files_keys,list):
        input_files_keys = [input_files_keys]
    temp_date_str = ''    
    for input_file_key in input_files_keys:
        input_files = data[input_file_key]
        if not isinstance(input_files,list):
            input_files = [input_files]
            for input_file in input_files:
                input_json = json_filename(input_file)
                if os.path.exists(input_json):
                    with open(input_json) as file:
                        input_data = json.load(file)
                    temp_date_str= max(input_data['date'], temp_date_str)
    data['input_modification_date'] = temp_date_str
    data['date']= actual_date.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, 'w') as file:
        json.dump(data, file, indent = 4)
        
def chek_same_existing_json(parameters, json_file,input_files_keys):
    with open(json_file) as file:
        prev_parameters = json.load(file)
    for key, value in parameters.items():
        #only comments can be different
        if key !='comments' and prev_parameters[key] != value:
            return False
    
    #Check dates for all input files dates are consistent
    for input_file_key in input_files_keys:
        input_files = parameters[input_file_key]

        if isinstance(input_files,str):
            input_files = [input_files] #generalize for input fields containing lists
        
        for input_file in  input_files:
            json_file = json_filename(input_file)
            if os.path.exists(json_file):
                with open(json_file) as in_file:
                    in_data = json.load(in_file)
                if prev_parameters['input_modification_date'] > in_data['date']:
                    return False
    return True


def same_json_or_remove(parameters:dict, input_files_keys:list,
                            output:str, verbose:bool)->bool:
    """
    If the file exist and the json is the same returns True. 
    Else: removes the file and the json associated with it.
    """
    json_file = json_filename(output)
    if os.path.exists(json_file):
        if os.path.exists(output):
            if chek_same_existing_json(parameters, json_file,input_files_keys):
                if verbose:
                    print(f"File {output} already created with these parameters")
                return True
        os.remove(json_file)
    if os.path.exists(output):
        os.remove(output)
    return False

def json_filename(filename:str)->str:
    return os.path.splitext(filename)[0]+'.json'

def del_key(d:dict,key:str)->dict:
    if key in d:
        copy_dict = d.copy()
        del copy_dict[key]
        return copy_dict
    return d