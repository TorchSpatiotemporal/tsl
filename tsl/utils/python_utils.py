import os
from typing import (
    Any,
    Sequence, List
)


def ensure_list(value: Any) -> List:
    # if isinstance(value, Sequence) and not isinstance(value, str):
    if hasattr(value, '__iter__') and not isinstance(value, str):
        return list(value)
    else:
        return [value]


def files_exist(files: Sequence[str]) -> bool:
    return len(files) != 0 and all([os.path.exists(f) for f in files])


def hash_dict(obj: dict):
    from hashlib import md5
    obj = {k: obj[k] for k in sorted(obj)}
    return md5(str(obj).encode()).hexdigest()


def set_property(obj, name, prop_function):
    """Add property :obj:`prop_function` to :obj:`obj`.

    :obj:`prop_function` must be a function taking only one argument, i.e.,
    :obj:`obj`.

    Args:
        obj (object): object on which the property has to be added.
        name (str): the name of the property.
        prop_function (function): function taking only :obj:`obj` as argument.
    """

    class_name = obj.__class__.__name__
    new_class = type(class_name, (obj.__class__,),
                     {name: property(prop_function)})
    obj.__class__ = new_class
