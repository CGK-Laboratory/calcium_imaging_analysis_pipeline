import pytest
from cgk_calcium_tools import isx_support
from cgk_calcium_tools.isx_aux_functions import create_random_movie


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