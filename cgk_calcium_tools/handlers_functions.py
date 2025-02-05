def get_deepest_level(hierarchy):
    groups = []
        
    current_level = [item for item in hierarchy if isinstance(item, str)]
    if current_level:
        groups.append(current_level) #list inside list
    
    for item in hierarchy:
        if isinstance(item, list):
            groups.extend(get_deepest_level(item))
            
    return groups


def flatt_lists(hierarchy):
    flat_list = []
    for element in hierarchy:
        if isinstance(element, list):
            flat_list.extend(flatt_lists(element))
        else:
            flat_list.append(element)
    return flat_list

def hierarqy_from_paths(recordings):
    root = {'.': []}
    for rec in recordings:
        parts = rec.name.split('/')
        current = root
        if len(parts) == 1:
            current['.'].append(rec)
        else:
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {'.': []}
                current = current[part]
            current['.'].append(rec)
    return _tree_to_list(root)

def _tree_to_list(node):
    result = []
    if '.' in node:
        result.extend(node['.'])
    for key in node:
        if key == '.':
            continue
        result.append(_tree_to_list(node[key]))
    return result

    
def update_nested_lists(nested_list, replace):
    for i in enumerate(nested_list):
        if isinstance(nested_list[i], list):
            update_nested_lists(nested_list[i], replace)
        else:
            if nested_list[i] in replace:
                nested_list[i] = replace[nested_list[i]]

def update_recordings(hierarchy, old_recs,new_recs):
    replace = {old:new for old,new in zip(old_recs,new_recs)}
    for i in enumerate(hierarchy):
        if isinstance(hierarchy[i], list):
            update_nested_lists(hierarchy, replace)
        else:
            if hierarchy[i] in replace:
                hierarchy[i] = replace[hierarchy[i]]