import os
import json
import isx
from datetime import datetime

def same_json_or_remove(parameters:dict, input_files_keys:list,
                            output:str, verbose:bool)->bool:
    """
    If the file exist and the json is the same returns True. 
    Else: removes the file and the json associated with it.
    """
    json_file = json_filename(output)
    if os.path.exists(json_file):
        if os.path.exists(output):
            if check_same_existing_json(parameters, json_file,input_files_keys,verbose):
                if verbose:
                    print(f"File {output} already created with these parameters")
                return True
        os.remove(json_file)
    if os.path.exists(output):
        os.remove(output)
    return False

def check_same_existing_json(parameters, json_file,input_files_keys,verbose):
    with open(json_file) as file:
        prev_parameters = json.load(file)
    for key, value in parameters.items():
        #only comments can be different
        if key !='comments' and prev_parameters[key] != value:
            if verbose:
                print(f'different {key}: old:{prev_parameters[key]}, new:{value}')
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
                    old_date = prev_parameters['input_modification_date']
                    new_date = in_data['date']
                    print(f'updated file {json_file}: old:{old_date}, new:{new_date}')
                    return False
    return True

def json_filename(filename:str)->str:
    return os.path.splitext(filename)[0]+'.json'

def remove_file_and_json(output):
    if os.path.exists(output):
        os.remove(output)
    json_file = json_filename(output)
    if os.path.exists(json_file):
        os.remove(json_file)

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