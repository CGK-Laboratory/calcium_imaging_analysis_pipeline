import unittest
import os
from cgk_calcium_tools.filehandler import isx_files_handler


class TestFunctions(unittest.TestCase):
    def test_from_local(self):
        my_path = os.getenv("CIA_INPUT_DATA")
        output = os.getenv("CIA_OUTPUT_DATA")
        all_env_vars = os.environ
        subfolder = [
            value
            for key, value in all_env_vars.items()
            if key.startswith("CIA_SUBFOLDER_")
        ]
        self.assertIsInstance(
            isx_files_handler(
                main_data_folder=my_path,
                outputsfolders=output,
                data_subfolders=subfolder,
                files_patterns="*.isxd",
                processing_steps=["PP", "TR", "BP", "MC"],
                one_file_per_folder=True,
                recording_labels=None,
                check_new_imputs=False,
                overwrite_metadata=False,
            ),
            isx_files_handler,
        )

    def test_from_nas(self):
        my_path = os.getenv("CIA_INPUT_DATA")
        output = os.getenv("CIA_OUTPUT_DATA")
        all_env_vars = os.environ
        subfolder = [
            value
            for key, value in all_env_vars.items()
            if key.startswith("CIA_SUBFOLDER_")
        ]
        self.assertIsInstance(
            isx_files_handler(
                main_data_folder=my_path,
                outputsfolders=output,
                data_subfolders=subfolder,
                files_patterns="*.isxd",
                processing_steps=["PP", "TR", "BP", "MC"],
                one_file_per_folder=False,
                recording_labels=None,
                check_new_imputs=True,
                overwrite_metadata=True,
            ),
            isx_files_handler,
        )


if __name__ == "__main__":
    unittest.main()
