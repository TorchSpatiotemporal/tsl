import inspect
from argparse import ArgumentParser
from typing import Set, Optional, Tuple

from torch import nn

from tsl.typing import ModelReturnOptions
from tsl.utils.python_utils import ensure_list


class BaseModel(nn.Module):
    r"""Base class for creating neural models.

    This class provides useful utilities for the model designer:

    * the methods :meth:`~tsl.nn.models.BaseModel.add_model_specific_args`
      and :meth:`~tsl.nn.models.BaseModel.add_argparse_args` allow to
      automatically add to an :class:`~argparse.ArgumentParser` the
      arguments needed to initialize the model (with typing and default
      values).

    * the method :meth:`~tsl.nn.models.BaseModel.loss` can be used to compute a
      custom loss on the provided training target. Inference modules in tsl will
      call this method for the loss computation, if implemented in the model.

    * the method :meth:`~tsl.nn.models.BaseModel.predict` can be used to define
      a variation of the :meth:`~torch.nn.Module.forward` function for only
      inference purposes (e.g., removing outputs used only for auxiliary tasks
      during training).

    * the parameter :attr:`return_type` specifies which the return type of the
      forward function (:class:`~torch.Tensor`, :obj:`list` or :obj:`dict`).
    """

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
        """Returns :obj:`True` if the model has implemented the
        :meth:`~tsl.nn.models.BaseModel.loss` method."""
        return self.loss.__qualname__.split('.')[0] != 'BaseModel'

    @property
    def has_predict(self) -> bool:
        """Returns :obj:`True` if the model has implemented the
        :meth:`~tsl.nn.models.BaseModel.predict` method."""
        return self.predict.__qualname__.split('.')[0] != 'BaseModel'

    def loss(self, target, *args, **kwargs):
        """Compute a custom loss w.r.t. :attr:`target`."""
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        """Forward function used only for inference."""
        return self.forward(*args, **kwargs)

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
    def model_excluded_args(cls) -> Set:
        """Set of arguments of :meth:`__init__` to be excluded when adding
        model's args to an :class:`~argparse.ArgumentParser` (see
        :meth:`~tsl.nn.models.BaseModel.add_model_specific_args`)."""
        return {'input_size', 'output_size', 'exog_size', 'n_nodes',
                'horizon', 'window'}

    @classmethod
    def add_model_specific_args(cls, parser: ArgumentParser):
        """Adds to the :class:`~argparse.ArgumentParser` :attr:`parser` the
        arguments needed to initialize the model (with typing and default
        values).

        The arguments added are all the parameters of the :meth:`__init__`
        method, excluding the keys returned by
        :meth:`~tsl.nn.models.BaseModel.model_excluded_args`."""
        return cls.add_argparse_args(parser,
                                     exclude_args=cls.model_excluded_args())

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser,
                          exclude_args: Optional[Set] = None):
        """Adds to the :class:`~argparse.ArgumentParser` :attr:`parser` all the
        parameters of the :meth:`__init__` method (with typing and default
        values)."""
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
