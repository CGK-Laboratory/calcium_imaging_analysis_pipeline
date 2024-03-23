# %%
import isx
from filehandler import isx_files_handler

# %%
# test_files = r"C:\Users\USER\Documents\Coti_lab\gretest"


# fx = isx_files_handler(
#     main_data_folder=r"E:\Gretana",
#     outputsfolders=r"C:\Users\USER\Documents\Coti_lab\gretest",
#     data_subfolders=[
#         "MUSC Rats-Cohort 1-D1-845D1-845_DB2_Day22_91621",
#     ],
#     files_patterns="*.isxd",
#     processing_steps=["PP", "TR", "BP", "MC"],
#     one_file_per_folder=False,
#     recording_labels=None,
#     check_new_imputs=True,
#     overwrite_metadata=True,
# )


# fx.de_interleave()

# %%
# fx.run_step("PP", temporal_downsample_factor=5, spatial_downsample_factor=10)

# %%
import isx
from filehandler import isx_files_handler


test = isx_files_handler(
    main_data_folder=r"C:\Users\USER\Documents\Coti_lab\gretest",
    outputsfolders=r"C:\Users\USER\Documents\Coti_lab\gretest\out",
    # data_subfolders=[], si se pone asi no tira error de no file found
    data_subfolders=["mi_nas"],
    files_patterns="*.isxd",
    processing_steps=["PP", "TR", "BP", "MC"],
    one_file_per_folder=False,
    recording_labels=None,
    check_new_imputs=True,
    overwrite_metadata=False,
)


test.de_interleave()

# %%

test.run_step("PP", temporal_downsample_factor=1, spatial_downsample_factor=1)

# %%
