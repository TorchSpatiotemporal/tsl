from typing import Any

from omegaconf import OmegaConf


def prod_resolver(*x):
    p = x[0]
    for elem in x[1:]:
        p *= elem
    return p


def ternary_resolver(cond: bool, truthy: Any, falsy: Any):
    return truthy if cond else falsy


def register_resolvers():
    # ${neg:-4} -> 4
    OmegaConf.register_new_resolver(name='neg', resolver=lambda x: -x)
    # ${in:2,[1,2,3]} -> True
    OmegaConf.register_new_resolver(name='in', resolver=lambda x, a: x in a)
    # ${not:True} -> False
    OmegaConf.register_new_resolver(name='not', resolver=lambda x: not x)
    # ${sum:1,2,3,4} -> 10
    OmegaConf.register_new_resolver(name='sum', resolver=lambda *x: sum(x))
    # ${prod:1,2,3,4} -> 24
    OmegaConf.register_new_resolver(name='prod', resolver=prod_resolver)
    # ${div:1,4} -> 0.25
    OmegaConf.register_new_resolver(name='div', resolver=lambda x, d: x / d)
    # ${exp:3,2} -> 9
    OmegaConf.register_new_resolver(name='exp', resolver=lambda x, e: x**e)
    # ${ternary:True,is_true,is_false} -> is_true
    OmegaConf.register_new_resolver(name='ternary', resolver=ternary_resolver)
