import tarfile
import zipfile

from tsl.utils import (
    ensure_list,
    extract_tar,
    extract_zip,
    files_exist,
    foo_signature,
    load_pickle,
    precision_stoi,
    save_pickle,
)
from tsl.utils.python_utils import (
    filter_kwargs,
    hash_dict,
    parameters_to_args,
    remove_files,
    set_property,
)


def test_list_files_signature_and_precision_helpers(tmp_path):
    path = tmp_path / 'file.txt'
    path.write_text('value')

    assert ensure_list('value') == ['value']
    assert ensure_list((1, 2)) == [1, 2]
    assert files_exist(path)
    assert not files_exist([path, tmp_path / 'missing'])
    assert foo_signature(lambda value, flag=False: None)['signature'] == [
        'value',
        'flag',
    ]
    assert precision_stoi('half') == 16
    assert precision_stoi(64) == 64


def test_pickle_and_archive_helpers(tmp_path):
    payload = {'answer': 42}
    pickle_path = save_pickle(payload, tmp_path / 'nested' / 'payload.pkl')
    assert load_pickle(pickle_path) == payload

    source = tmp_path / 'source.txt'
    source.write_text('content')
    zip_path = tmp_path / 'archive.zip'
    with zipfile.ZipFile(zip_path, 'w') as archive:
        archive.write(source, 'source.txt')
    tar_path = tmp_path / 'archive.tar'
    with tarfile.open(tar_path, 'w') as archive:
        archive.add(source, arcname='source.txt')

    zip_output, tar_output = tmp_path / 'zip', tmp_path / 'tar'
    extract_zip(zip_path, zip_output, log=False)
    extract_tar(tar_path, tar_output, log=False)
    assert (zip_output / 'source.txt').read_text() == 'content'
    assert (tar_output / 'source.txt').read_text() == 'content'


def test_python_utility_helpers(tmp_path):
    class Sample:
        pass

    sample = Sample()
    set_property(sample, 'answer', lambda _: 42)
    assert sample.answer == 42
    assert hash_dict({'b': 2, 'a': 1}) == hash_dict({'a': 1, 'b': 2})

    def target(required: int, optional=1):
        return required + optional

    parser = parameters_to_args(target)
    assert parser.parse_args(['--required', '2']).required == 2
    assert filter_kwargs(target, {'required': 1, 'extra': 2}) == {'required': 1}

    checkpoint = tmp_path / 'model.ckpt'
    other = tmp_path / 'model.txt'
    checkpoint.touch()
    other.touch()
    remove_files(tmp_path)
    assert not checkpoint.exists()
    assert other.exists()
