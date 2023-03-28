import inspect
import os
from argparse import ArgumentParser
from typing import (Any, Callable, List, Mapping, Optional, Sequence, Set,
                    Type, Union)


def ensure_list(value: Any) -> List:
    # if isinstance(value, Sequence) and not isinstance(value, str):
    if hasattr(value, '__iter__') and not isinstance(value, str):
        return list(value)
    else:
        return [value]


def files_exist(files: Sequence[str]) -> bool:
    files = ensure_list(files)
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
    new_class = type(class_name, (obj.__class__, ),
                     {name: property(prop_function)})
    obj.__class__ = new_class


def foo_signature(foo: Union[Callable, Type]):
    if isinstance(foo, type):
        foo = foo.__init__
    argspec = inspect.getfullargspec(foo)
    args = argspec.args
    if len(args) and args[0] in ['self', 'cls']:  # temp, to do better
        args = args[1:]
    has_args = argspec.varargs is not None
    has_kwargs = argspec.varkw is not None
    return {'signature': args, 'has_args': has_args, 'has_kwargs': has_kwargs}


def parameters_to_args(foo: Union[Callable, Type],
                       parser: Optional[ArgumentParser] = None,
                       exclude_args: Optional[Set] = None):
    if isinstance(foo, type):
        foo = foo.__init__
    sign = inspect.signature(foo)
    # filter excluded arguments
    excluded = {'self'}
    if exclude_args is not None:
        excluded.update(exclude_args)
    # filter excluded arguments
    if parser is None:
        parser = ArgumentParser()
    # parse signature
    for name, param in sign.parameters.items():
        if name in excluded:
            continue
        name = '--' + name.replace('_', '-')
        kwargs = dict()
        if param.annotation is not inspect._empty:
            kwargs['type'] = param.annotation
        if param.default is not inspect._empty:
            kwargs['default'] = param.default
            if 'type' not in kwargs:
                kwargs['type'] = type(param.default)
        try:
            parser.add_argument(name, **kwargs)
        except ValueError:
            if 'default' in kwargs:
                parser.add_argument(name, default=kwargs['default'])
            else:
                parser.add_argument(name)
    return parser


def precision_stoi(precision: Union[int, str]) -> int:
    """Return precision as int if expressed as a string. Allowed strings are
    :obj:`half`=16, :obj:`full`=32, :obj:`double`=64."""
    if isinstance(precision, str):
        precision = dict(half=16, full=32, double=64).get(precision)
    assert precision in [16, 32, 64], \
        "precision must be one of 16 (or 'half'), 32 (or 'full') or 64 " \
        f"(or 'double'). Default is 32, invalid input '{precision}'."
    return precision


def remove_files(directory: str, extension: str = '.ckpt'):
    """Remove files of specific extension from a directory"""
    files_in_directory = os.listdir(directory)
    filtered_files = [
        file for file in files_in_directory if file.endswith(extension)
    ]
    for file in filtered_files:
        path_to_file = os.path.join(directory, file)
        os.remove(path_to_file)


def filter_kwargs(target: Union[Callable, Type], kwargs: Mapping):
    """
    Filters a dictionary to match the signature of an input class or function.

    Args:
        target: The target class or function.
        kwargs: The dictionary to filter.

    Returns:
        The filtered dictionary.
    """
    signature = foo_signature(target)
    if not signature['has_kwargs']:
        kwargs = {
            k: v
            for k, v in kwargs.items() if k in signature['signature']
        }
    return kwargs
