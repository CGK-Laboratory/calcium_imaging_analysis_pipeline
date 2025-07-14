def get_parent_and_file(d,key, value):
    """
    Search for a child element in a .isxp project with the given key and value
    and return the file name of the parent element and the file name of that element.

    Parameters
    ----------
    d : dict
        Part of a .isxp file
    key : str
        Key to search for
    value : str
        Value to search for
    
    Returns
    -------
    tuple
        A tuple of two strings. The first one is the file name of the parent
        element, and the second one is the file name of the child element.
    """
    if not isinstance(d, dict):
        return None
    child_found = searched_child(d,key, value) 
    if child_found is not None:
        return (extract_file(d),child_found)
    for child in d:
        if isinstance(d[child], dict):
            result = get_parent_and_file(d[child],key, value)
            if result is not None:
                return result
        if isinstance(d[child], list):
            for ch in d[child]:
                result = get_parent_and_file(ch,key, value)
                if result is not None:
                    return result
    return None

def searched_child(d,key, value):
    """
    Search for a child element in a .isxp project with the given key and value
    and return the file name of that element.
    
    Parameters
    ----------
    d : dict
        Part of a .isxp file
    key : str
        Key to search for
    value : str
        Value to search for
    
    Returns
    -------
    str
        The file name of the child element or None if not found
    """
    if isinstance(d, dict) and 'children' in d.keys():
        for child in d['children']:
            if key in child and child[key]==value:
                return extract_file(child)
    return None

def extract_file(d:dict):
    """
    Extract the file name from a .isxp element
    
    Parameters
    ----------
    d : dict
        part of a.isxp file
        
    Raises
    ------
    AssertionError
        If there is more than one dataSets element
    """
    assert len(d['dataSets'])==1,'Error more than one file'+str(d['dataSets'])
    return d['dataSets'][0]['fileName']