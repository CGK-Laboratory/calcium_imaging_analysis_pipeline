import unittest
import filehandler
import longitudinal_registration
import os


class TestFunctions(unittest.TestCase):
    def test_from_local(self):
        test_files = r"C:\Users\USER\Documents\Coti_lab"
        self.assertIsInstance(
            filehandler.isx_files_handler(
                main_data_folder="C:\Program Files\Inscopix\Data Processing",
                outputsfolders=test_files,
                data_subfolders=".",
                files_patterns=".isx",
                processing_steps=["PP", "TR", "BP", "MC"],
                one_file_per_folder=True,
                recording_labels=None,
                check_new_imputs=False,
                overwrite_metadata=True,
            ),
            filehandler.isx_files_handler,
        )

    def test_longitudinal_registration(self):
        test_cell_set = [
            r"C:\Program Files\Inscopix\Data Processing\isx\examples\multicolor-registration\demo-data\red-cellset.isxd",
            r"C:\Program Files\Inscopix\Data Processing\isx\examples\multicolor-registration\demo-data\green-cellset.isxd",
        ]
        test_output_cell_set = [
            r"C:\Users\USER\Documents\Coti_lab\calcium_imaging_analysis_pipeline\cgk_calcium_tools\casa.isxd",
            r"C:\Users\USER\Documents\Coti_lab\calcium_imaging_analysis_pipeline\cgk_calcium_tools\greta.isxd",
        ]
        assert longitudinal_registration.longitudinal_registration(
            test_cell_set, test_output_cell_set, "test_long_reg.json"
        )
        for i in test_output_cell_set:
            os.remove(i)
        os.remove(
            r"C:\Users\USER\Documents\Coti_lab\calcium_imaging_analysis_pipeline\cgk_calcium_tools\test_long_reg.csv"
        )
        os.remove(
            r"C:\Users\USER\Documents\Coti_lab\calcium_imaging_analysis_pipeline\cgk_calcium_tools\test_long_reg.json"
        )


if __name__ == "__main__":
    unittest.main()
