import inspect
from argparse import ArgumentParser
from typing import Set, Optional, Tuple

from torch import nn

from tsl.typing import ModelReturnOptions
from tsl.utils.python_utils import ensure_list


class BaseModel(nn.Module):

    def __init__(self, return_type: ModelReturnOptions = None):
        super(BaseModel, self).__init__()
        self.return_type = return_type
        if return_type is not None:
            self.register_forward_hook(self.forward_packer)
        self.forward_signature, \
        self.has_forward_args, \
        self.has_forward_kwargs = self._get_forward_signature()

    @property
    def has_loss(self) -> bool:
        return self.loss.__qualname__.split('.')[0] != 'BaseModel'

    def loss(self, target, *args, **kwargs):
        raise NotImplementedError

    def _get_forward_signature(self) -> Tuple[list, bool, bool]:
        has_args, has_kwargs = False, False
        fwd_params = inspect.signature(self.forward).parameters
        args = []
        for name, param in fwd_params.items():
            if param.kind == inspect._ParameterKind.VAR_POSITIONAL:
                has_args = True
            elif param.kind == inspect._ParameterKind.VAR_KEYWORD:
                has_kwargs = True
            elif name != 'self':
                args.append(name)
        return args, has_args, has_kwargs

    def forward_packer(self, input, output):
        if isinstance(output, self.return_type):
            return output
        if self.return_type is list:
            return ensure_list(output)
        raise TypeError(f"return type of forward ({type(output)}) does not "
                        f"match with {self.__class__.__name__}.return_type "
                        f"({self.return_type}).")

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser,
                          exclude_args: Optional[Set] = None):
        sign = inspect.signature(cls.__init__)
        # filter excluded arguments
        excluded = {'self'}
        if exclude_args is not None:
            excluded.update(exclude_args)
        # parse signature
        for name, param in sign.parameters.items():
            if name in exclude_args:
                continue
            name = '--' + name.replace('_', '-')
            kwargs = dict()
            if param.annotation is not inspect._empty:
                kwargs['type'] = param.annotation
            if param.default is not inspect._empty:
                kwargs['default'] = param.default
                if 'type' not in kwargs:
                    kwargs['type'] = type(param.default)
            parser.add_argument(name, **kwargs)
        return parser

    @classmethod
    def model_excluded_args(cls) -> Set:
        return set()

    @classmethod
    def add_model_specific_args(cls, parser: ArgumentParser):
        exclude_args = {'input_size', 'output_size', 'exog_size',
                        'n_nodes', 'horizon', 'window'}
        exclude_args.update(cls.model_excluded_args())
        return cls.add_argparse_args(parser, exclude_args=exclude_args)
