"""Downloaded bundles preserve artifact bindings and reject damaged/unsafe archives."""

import hashlib
import json
import zipfile

import pytest

from trustworthy_recsys.retrieval.bundle import pack, unpack


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bundle_roundtrip_and_checksum_failure(tmp_path):
    inputs, model = tmp_path/'inputs', tmp_path/'model'
    inputs.mkdir(); model.mkdir()
    (inputs/'item_ids.json').write_text('["1", "2"]')
    (inputs/'manifest.json').write_text(json.dumps({'outputs': {'item_ids.json': digest(inputs/'item_ids.json')}}))
    (model/'training.json').write_text(json.dumps({'input_manifest_sha256': digest(inputs/'manifest.json')}))
    (model/'manifest.json').write_text(json.dumps({'outputs': {'training.json': digest(model/'training.json')}}))
    note = tmp_path/'note.txt'
    note.write_text('Synthetic test documentation')
    bundle = tmp_path/'download.zip'
    pack(inputs, model, bundle, note, note, note, note)
    output = tmp_path/'restored'
    unpack(bundle, output)
    assert (output/'inputs/manifest.json').read_bytes() == (inputs/'manifest.json').read_bytes()
    assert (output/'model/training.json').read_bytes() == (model/'training.json').read_bytes()
    with pytest.raises(ValueError, match='already exists'):
        unpack(bundle, output)
    with bundle.open('ab') as handle:
        handle.write(b'corruption')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        unpack(bundle, tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_bundle_rejects_path_traversal_before_extraction(tmp_path):
    bundle = tmp_path/'unsafe.zip'
    with zipfile.ZipFile(bundle,'w') as archive:
        archive.writestr('../escaped.txt','bad')
    bundle.with_suffix('.zip.sha256').write_text(digest(bundle)+'  unsafe.zip\n')
    with pytest.raises(ValueError, match='Unsafe archive path'):
        unpack(bundle,tmp_path/'restore')
    assert not (tmp_path/'escaped.txt').exists()
