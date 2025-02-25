import pytest
from cgk_calcium_tools.isx_aux_functions import load_isxd_files
from cgk_calcium_tools.isx_aux_functions import create_random_movie
from pathlib import Path
@pytest.fixture(scope="module")
def tmp_test_path(tmp_path_factory):
    """Create a shared temporary directory at session scope."""
    path = tmp_path_factory.mktemp("test_data")

    subfolders = ['animal_A','animal_B']
    subsubfolders = ['task1','task2']
    for s in subfolders:
        for ss in subsubfolders:
            (path / s / ss).mkdir(parents=True)
            create_random_movie(output_file=str(path / s / ss / 'random_movie.isxd'),
                                num_samples=500, num_pixels=(30, 40))
    return str(path)

@pytest.mark.skipif(load_isxd_files is None, reason="ISX library not available")
def test_load_isxd_files(tmp_test_path):
    fh=load_isxd_files(tmp_test_path,tmp_test_path,infere_hierarchy=True)
    assert len(fh.recordings_list)==4

    subfolders = ['animal_A','animal_B']
    subsubfolders = ['task1','task2']
    files_hierarchy = []
    for s in subfolders:
        sublist = []
        for ss in subsubfolders:
            sublist.append((str(Path(s) / ss/ 'random_movie.isxd')))
        files_hierarchy.append(sublist)
    files_checkd = 0 
    for folders in fh.recordings:
        current_folder = None
        for folder in folders:
            for rec in folder:
                if current_folder is None:
                    if rec.file in files_hierarchy[0]:
                        current_folder = 0
                    else:
                        current_folder = 1
            if rec.file in files_hierarchy[current_folder]:
                files_checkd += 1
    assert files_checkd == 4