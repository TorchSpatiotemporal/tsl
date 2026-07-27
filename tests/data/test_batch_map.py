"""Unit tests for :mod:`tsl.data.batch_map` (BatchMapItem, BatchMap)."""
import pytest

from tsl.data.batch_map import BatchMap, BatchMapItem
from tsl.data.synch_mode import HORIZON, STATIC, WINDOW, SynchMode


# -- BatchMapItem -----------------------------------------------------------

def test_keys_coerced_to_list():
    assert BatchMapItem('target').keys == ['target']
    assert BatchMapItem(['a', 'b']).keys == ['a', 'b']


def test_multiple_keys_require_cat_dim():
    # several keys with cat_dim=None is ambiguous and must fail
    with pytest.raises(RuntimeError):
        BatchMapItem(['a', 'b'], cat_dim=None)
    # a single key with cat_dim=None is fine
    assert BatchMapItem('a', cat_dim=None).cat_dim is None
    # multiple keys with an explicit cat_dim is fine
    assert BatchMapItem(['a', 'b'], cat_dim=0).cat_dim == 0


def test_synch_mode_from_string_and_enum():
    assert BatchMapItem('a', synch_mode='window').synch_mode is WINDOW
    assert BatchMapItem('a', synch_mode='HORIZON').synch_mode is HORIZON
    assert BatchMapItem('a', synch_mode=STATIC).synch_mode is STATIC


def test_shape_coerced_to_tuple():
    item = BatchMapItem('a', shape=[2, 3])
    assert item.shape == (2, 3)
    assert BatchMapItem('a').shape is None


def test_kwargs_returns_dict():
    item = BatchMapItem('a', cat_dim=-1, pattern='t n f')
    kw = item.kwargs()
    assert kw is item.__dict__
    # the dict can be splatted back into a fresh item
    assert BatchMapItem(**kw).keys == ['a']


# -- synch_mode auto-inference from pattern ---------------------------------

def test_pattern_with_t_infers_window():
    assert BatchMapItem('a', pattern='t n f').synch_mode is WINDOW


def test_pattern_without_t_infers_static():
    assert BatchMapItem('a', pattern='n f').synch_mode is STATIC


def test_explicit_synch_mode_not_overridden_by_pattern():
    item = BatchMapItem('a', synch_mode=HORIZON, pattern='t n f')
    assert item.synch_mode is HORIZON


def test_repr_smoke():
    r = repr(BatchMapItem('a', pattern='t n f', shape=(1, 2)))
    assert 'a' in r and 't n f' in r


# -- BatchMap ---------------------------------------------------------------

def test_setitem_type_coercion():
    bm = BatchMap()
    bm['from_item'] = BatchMapItem('a')
    bm['from_str'] = 'b'
    bm['from_list'] = ['c', 'd']
    bm['from_tuple'] = (['e', 'f'], 'window', True, 0)
    bm['from_mapping'] = dict(keys='g', pattern='t n f')
    for key in bm:
        assert isinstance(bm[key], BatchMapItem)
    assert bm['from_str'].keys == ['b']
    assert bm['from_list'].keys == ['c', 'd']
    assert bm['from_tuple'].synch_mode is WINDOW
    assert bm['from_mapping'].pattern == 't n f'


def test_setitem_invalid_type_raises():
    bm = BatchMap()
    with pytest.raises(TypeError):
        bm['bad'] = 42


def test_mapping_interface():
    bm = BatchMap(a='x', b='y')
    assert len(bm) == 2
    assert set(iter(bm)) == {'a', 'b'}
    assert bm['a'].keys == ['x']


def test_update():
    bm = BatchMap(a='x')
    bm.update(b='y', c=['z1', 'z2'])
    assert set(bm) == {'a', 'b', 'c'}


def test_by_synch_mode_filters():
    bm = BatchMap()
    bm['win'] = BatchMapItem('a', pattern='t n f')      # WINDOW
    bm['stat'] = BatchMapItem('b', pattern='n f')        # STATIC
    bm['hor'] = BatchMapItem('c', synch_mode=HORIZON)
    assert set(bm.by_synch_mode(WINDOW)) == {'win'}
    assert set(bm.by_synch_mode(STATIC)) == {'stat'}
    assert set(bm.by_synch_mode(HORIZON)) == {'hor'}
    assert bm.by_synch_mode(SynchMode.WINDOW)['win'] is bm['win']


def test_repr_smoke_batchmap():
    bm = BatchMap(a='x')
    assert 'BatchMap' in repr(bm) and 'a' in repr(bm)
