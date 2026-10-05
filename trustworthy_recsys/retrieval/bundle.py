"""Package and verify portable trained-model downloads without retraining."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

from trustworthy_recsys.data.retrieval_inputs import read_json
from trustworthy_recsys.data.reporting import hashes


def artifact_files(root):
    root = Path(root)
    if (root/'INCOMPLETE').exists():
        raise ValueError(f'Incomplete artifacts: {root}')
    manifest = read_json(root/'manifest.json')
    paths = [root/'manifest.json']
    for name, expected in manifest['outputs'].items():
        if Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError('Only flat artifact manifests are supported')
        path = root/name
        if hashes(path)['sha256'] != expected:
            raise ValueError(f'Artifact hash mismatch: {path}')
        paths.append(path)
    return paths


def pack(inputs_dir, model_dir, output, instructions, source_readme, source_card, model_card):
    inputs, model, output = Path(inputs_dir), Path(model_dir), Path(output)
    checksum = output.with_suffix(output.suffix+'.sha256')
    if output.exists() or checksum.exists():
        raise ValueError('Choose a new bundle filename; existing bundles are never overwritten')
    payload = {}
    for prefix, root in [('inputs', inputs), ('model', model)]:
        for path in artifact_files(root):
            payload[f'{prefix}/{path.name}'] = path
    training = read_json(model/'training.json')
    if training['input_manifest_sha256'] != hashes(inputs/'manifest.json')['sha256']:
        raise ValueError('Model and input manifests do not match')
    for name, path in [('README.md', instructions), ('MOVIELENS_README.txt', source_readme),
                       ('SOURCE_DATA_CARD.md', source_card), ('MODEL_CARD.md', model_card)]:
        payload[name] = Path(path)
    manifest = {'format_version': 1, 'purpose': 'Trained two-tower model and complete prepared inputs; no retraining required',
                'input_manifest_sha256': hashes(inputs/'manifest.json')['sha256'],
                'model_manifest_sha256': hashes(model/'manifest.json')['sha256'],
                'files': {name: hashes(path)['sha256'] for name, path in sorted(payload.items())}}
    output.parent.mkdir(parents=True, exist_ok=True)
    # ZIP entries have fixed timestamps and portable relative names.
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(payload.items()):
            info = zipfile.ZipInfo(name, date_time=(1980,1,1,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            with path.open('rb') as source, archive.open(info, 'w', force_zip64=True) as target:
                shutil.copyfileobj(source, target, length=1024*1024)
        archive.writestr('bundle_manifest.json', json.dumps(manifest, indent=2)+'\n')
    checksum.write_text(f"{hashes(output)['sha256']}  {output.name}\n", encoding='utf-8')
    print(f'Created {output}: {output.stat().st_size / 1024**2:.1f} MiB, {len(payload)} payload files', flush=True)
    return manifest


def unpack(bundle, output_dir, checksum_file=None):
    bundle, out = Path(bundle), Path(output_dir).resolve()
    if out.exists():
        raise ValueError('Extraction directory already exists; choose a new directory')
    checksum = Path(checksum_file) if checksum_file else bundle.with_suffix(bundle.suffix+'.sha256')
    expected = checksum.read_text(encoding='utf-8').split()[0]
    if hashes(bundle)['sha256'] != expected:
        raise ValueError('Bundle SHA-256 mismatch; download the complete Git LFS object')
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive entries')
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
                raise ValueError('Unsafe archive path')
            if not (out/name).resolve().is_relative_to(out):
                raise ValueError('Archive path escapes output directory')
        manifest = json.loads(archive.read('bundle_manifest.json'))
        if manifest.get('format_version') != 1 or set(names) != set(manifest['files']) | {'bundle_manifest.json'}:
            raise ValueError('Bundle manifest does not match archive contents')
        # Verify the full archive before writing any extracted files.
        for name, expected in manifest['files'].items():
            digest = hashlib.sha256()
            with archive.open(name) as source:
                for chunk in iter(lambda: source.read(1024*1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f'Bundled file hash mismatch: {name}')
        out.mkdir(parents=True)
        (out/'INCOMPLETE').write_text('Bundle extraction in progress\n')
        archive.extractall(out)
        (out/'INCOMPLETE').unlink()
    print(f'Verified and extracted bundle to {out}', flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('pack')
    for name in ('inputs-dir','model-dir','output','instructions','source-readme','source-card','model-card'):
        build.add_argument('--'+name, type=Path, required=True)
    extract = commands.add_parser('unpack')
    extract.add_argument('--bundle', type=Path, required=True)
    extract.add_argument('--output-dir', type=Path, required=True)
    extract.add_argument('--checksum-file', type=Path)
    args = vars(parser.parse_args())
    command = args.pop('command')
    (pack if command == 'pack' else unpack)(**args)


if __name__ == '__main__':
    main()
