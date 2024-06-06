import ast
import copy
import inspect
import json
import math
import os
import re
import types
from typing import Any, List, Union, Tuple

import pandas as pd
import yaml

__all__ = ["IgnorePrintTransformer", "load_json", "load_yaml", "save_json", "save_yaml",
           "compare_structures", "get_changes", "ast_to_json", "get_repo_url_details", "flatten_dict"]


class IgnorePrintTransformer(ast.NodeTransformer):
    def visit_Expr(self, node: ast.Expr) -> Any:
        # Check if the expression is a function call to 'print'
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'print':
            return None  # Remove the node
        return node  # Otherwise, return the node unchanged


def load_json(filename: str) -> dict:
    """
    Load a JSON file into a dictionary.

    Parameters
    ----------
    filename : str
        The filename to load the JSON file from.

    Returns
    -------
    dict
        The dictionary loaded from the JSON file.
    """
    with open(os.path.join(filename), "r") as file:
        return json.load(file)


def load_yaml(filename: str) -> dict:
    """
    Load a YAML file into a dictionary.

    Parameters
    ----------
    filename : str
        The filename to load the YAML file from.

    Returns
    -------
    dict
        The dictionary loaded from the YAML file.
    """
    with open(os.path.join(filename), "r") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def save_json(data: dict, filename: str) -> None:
    """
    Save a dictionary to a JSON file.

    Parameters
    ----------
    data : dict
        The dictionary to save.

    filename : str
        The filename to save the dictionary to.
    """
    with open(os.path.join(filename), "w") as file:
        file.write(json.dumps(data, indent=4, sort_keys=False))


def save_yaml(data: dict, filename: str) -> None:
    """
    Save a dictionary to a YAML file.

    Parameters
    ----------
    data : dict
        The dictionary to save.
    filename : str
        The filename to save the dictionary to.
    """
    with open(os.path.join(filename), "w") as file:
        yaml.dump(data, file)


def save_csv(data: pd.DataFrame, filename: str) -> None:
    """
    Save a dictionary to a YAML file.

    Parameters
    ----------
    data : pd.DataFrame
        The dictionary to save.
    filename : str
        The filename to save the dictionary to.
    """
    data.to_csv(filename, index=False)


def are_both_nan_or_inf(a: Any, b: Any) -> bool:
    """
    Check if both values are NaN or infinity.

    Parameters
    ----------
    a : Any
        The first value to compare.
    b : Any
        The second value to compare.

    Returns
    -------
    bool
        True if both values are NaN or infinity, False otherwise.
    """
    if isinstance(a, (int, float,)) and isinstance(b, (int, float,)):
        return math.isnan(a) and math.isnan(b) or math.isinf(a) and math.isinf(b)
    return False


def compare_structures(dict1: dict, dict2: dict, path: str = '') -> List[str]:
    """
    Compare two dictionaries and return the differences between them.

    Parameters
    ----------
    dict1 : dict
        The first dictionary to compare.


    dict2 :
        The second dictionary to compare.
    path :
        The path to the current dictionary, used for recursive calls.

    Returns
    -------
    List[str]
        A list of differences between the two dictionaries.
    """
    differences = []

    starting_path = copy.deepcopy(path)

    # Check for keys missing in the second dict
    for key in dict1:
        path = f"{path}.{key}" if path else key
        if key not in dict2:
            differences.append(f"Removed {path}")
        else:
            value1 = dict1[key]
            value2 = dict2[key]

            # If both are dicts, recurse. If not, compare values directly.
            if isinstance(value1, dict) and isinstance(value2, dict):
                differences.extend(compare_structures(value1, value2, ""))
            elif value1 != value2:
                if not are_both_nan_or_inf(value1, value2):
                    differences.append(f"Changed {path} from {value1} to {value2}")

    # Check for keys added in the second dict
    for key in dict2:
        starting_path = f"{starting_path}.{key}" if starting_path else key
        if key not in dict1:
            differences.append(f"Added {starting_path} with value {dict2[key]}")

    return differences


def get_changes(dict1: dict, dict2: dict, path: str = '') -> List[str]:
    """
    Get the changes between two dictionaries.

    Parameters
    ----------
    dict1 : dict
        The first dictionary to compare.
    dict2 :
        The second dictionary to compare.
    path :
        The path to the current dictionary, used for recursive calls.

    Returns
    -------
    List[str]
        A list of changes between the two dictionaries.

    """
    changes = []

    # Check for changes and additions
    for key, value in dict2.items():
        current_path = f"{path}.{key}" if path else str(key)
        if key not in dict1:
            changes.append(f"Added {current_path}")
        elif dict1[key] != value:
            if isinstance(value, dict) and isinstance(dict1[key], dict):
                changes.extend(get_changes(dict1[key], value, current_path))
            elif not are_both_nan_or_inf(dict1[key], value):
                changes.append(f"changed {current_path} from {dict1[key]} to {value}")

    # Check for removals
    for key in dict1:
        if key not in dict2:
            current_path = f"{path}.{key}" if path else str(key)
            changes.append(f"Removed {current_path}")

    return changes


def ast_to_json(node: ast.AST) -> Union[dict, list, Any]:
    """
    Converts an AST node to a JSON-serializable dictionary.

    Parameters
    ----------
    node : ast.AST
        The AST node to convert.

    Returns
    -------
    dict
        The JSON-serializable dictionary.
    """

    if isinstance(node, ast.AST):
        node_dict = {"_type": node.__class__.__name__}
        for field in node._fields:
            node_dict[field] = ast_to_json(getattr(node, field))
        return node_dict
    elif isinstance(node, list):
        return [ast_to_json(n) for n in node]
    else:
        return node


def is_valid_repo_url(url: str) -> bool:
    """
    Checks if a URL is a valid repository URL. Only supports GitHub, GitLab, and Bitbucket repositories.

    Parameters
    ----------
    url : str
        The URL to validate.

    Returns
    -------
    bool
        True if the URL is valid, False otherwise.

    """
    pattern = r'^https://(github\.com|gitlab\.com|bitbucket\.org)/[\w.-]+/[\w.-]+(?:\.git)?$'
    return bool(re.match(pattern, url))


def serialize(item: Any) -> dict:
    """
    Serializes an object into a dictionary.

    Parameters
    ----------
    item : Any
        The object to serialize.

    Returns
    -------
    dict
        The serialized object.
    """

    result = {}

    if item is None:
        return result
    if hasattr(item, '__dict__'):  # it's an object
        result['type'] = type(item).__name__
        for attr in vars(item):
            value = getattr(item, attr)
            result[attr] = serialize(value)
    elif isinstance(item, dict):  # it's a dictionary
        for key, value in item.items():

            if not key.startswith("__") \
                    and not isinstance(value, types.ModuleType) \
                    and not isinstance(value, types.FunctionType) \
                    and not isinstance(value, type):
                result[key] = serialize(value)
    elif isinstance(item, (list, tuple, set)):  # it's a list
        result = [serialize(value) for value in item]
    else:  # it's a basic data type
        return item

    return result


def get_repo_url_details(url: str) -> Tuple[Union[str, Any]]:
    """
    Extracts the domain (GitHub, GitLab, Bitbucket), owner, and repository name from a repository URL.

    Parameters
    ----------
    url : str
        The repository URL to extract details from.

    Returns
    -------
    tuple
        A tuple containing the domain, owner, and repository name.
    """
    pattern = re.compile(r'https?://(?:[^@]+@)?(github|gitlab|bitbucket)\.[^/]+/([^/]+)/([^/.]+)')
    match = pattern.match(url)
    if match:
        return match.groups()
    else:
        raise AttributeError("Invalid repository URL")


def flatten_dict(d: dict, parent_key: str = '') -> dict:
    """
    Flattens a nested dictionary.

    Parameters
    ----------
    d : dict
        The dictionary to flatten
    parent_key : str, optional
        The parent key of the dictionary, used for recursive calls.

    Returns
    -------
    dict
        A flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}__{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_caller_path():
    """
    Change the current working directory to the directory of the caller script.
    """

    path = os.path.abspath("")

    if not is_running_in_notebook():
        # Get the current frame
        current_frame = inspect.currentframe()

        # Traverse back through the call stack until we find a frame that is not in this module
        calling_frame = current_frame
        while calling_frame:
            calling_frame = calling_frame.f_back
            # If the file name of the current frame is different from this file, we found the calling script
            if calling_frame and calling_frame.f_code.co_filename != __file__:
                # Further checks to ensure it's outside a class' __init__ if needed
                # Check if we are in the __init__ function of a class
                if calling_frame.f_code.co_name != '__init__' and calling_frame.f_code.co_name != "wrapper":
                    path = os.path.dirname(os.path.abspath(calling_frame.f_code.co_filename))

    return path


def is_running_in_notebook() -> bool:
    """
    Check if the code is running in a Jupyter notebook.

    Returns
    -------
    bool
        True if running in a Jupyter notebook, False otherwise.
    """
    from IPython import get_ipython

    ipython = get_ipython()

    if ipython is None:
        return False

    return 'IPKernelApp' in ipython.config
