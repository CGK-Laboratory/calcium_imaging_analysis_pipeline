## How to run the tests

To run the tests located in test_init.py, you need to set environment variables to configure the input and output folders. If a folder is not required, it is not necessary to set it.

The full path of the main folder should be an enviroment variable `CIA_INPUT_DATA`, the outputfolder full path should be an enviroment variable as well as `CIA_OUTPUT_DATA`, and the subfolders names in variables of the form `CIA_SUBFOLDER_...` followed by numbers or any other way to indicate the different subfolders.


For example for CMD in Windows, you must perform the setting as follows:

    set CIA_INPUT_DATA=C:\path\raw\data    
    set CIA_OUTPUT_DATA=C:\path\output\data   
    set CIA_SUBFOLDER_1=subfolder_data_1
    set CIA_SUBFOLDER_2=subfolder_data_2
    .
    .
    .
    set CIA_SUBFOLDER_n=subfolder_data_n







