import pytest
isx = pytest.importorskip("isx")
from cgk_calcium_tools.isx_aux_functions import load_isxd_files

def test_isx_processing(tmp_test_path):
    fh = load_isxd_files(tmp_test_path,tmp_test_path,infere_hierarchy=True)
    preprocessed = fh.apply('isx:preprocessing')
    return preprocessed