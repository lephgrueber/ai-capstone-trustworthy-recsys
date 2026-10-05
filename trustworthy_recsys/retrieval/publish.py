"""Publish a trained model's embeddings to a new versioned Weaviate collection."""

import argparse
from pathlib import Path
import shutil

import numpy as np

from trustworthy_recsys.data.retrieval_inputs import ModelInputs, read_json
from trustworthy_recsys.data.reporting import hashes, write_json, code_provenance
from .model import load_model
from .weaviate_store import WeaviateIndex


def publish(inputs_dir, model_dir, output_dir, collection):
    source, out = Path(model_dir), Path(output_dir)
    if out.exists():
        raise ValueError('Published model directory already exists; choose a new directory')
    if (source/'INCOMPLETE').exists():
        raise ValueError('Source model is incomplete')
    inputs = ModelInputs(inputs_dir, verify=True)
    training = read_json(source/'training.json')
    if training['input_manifest_sha256'] != hashes(Path(inputs_dir)/'manifest.json')['sha256']:
        raise ValueError('Model was trained with different inputs')
    for name, digest in read_json(source/'manifest.json')['outputs'].items():
        if hashes(source/name)['sha256'] != digest:
            raise ValueError(f'Model artifact hash mismatch: {name}')
    vectors = np.load(source/'item_embeddings.npy')
    model = load_model(source/'model.pt', inputs.item_features)
    if not np.allclose(vectors, model.all_item_vectors(), atol=1e-6):
        raise ValueError('Saved embeddings do not match the selected model')
    out.mkdir(parents=True)
    (out/'INCOMPLETE').write_text('Weaviate publication in progress\n')
    for name in ('model.pt', 'item_embeddings.npy', 'training_pair_indices.npy', 'validation_user_ids.json'):
        shutil.copyfile(source/name, out/name)
    index = WeaviateIndex.create(collection, inputs.item_ids, vectors)
    try:
        write_json(out/'weaviate.json', index.descriptor)
        training['source_training_index'] = training.get('source_training_index', training.get('index'))
        training['index'] = index.descriptor['index']
        training['publication'] = {'source_model_manifest_sha256': hashes(source/'manifest.json')['sha256'],
                                   'code': code_provenance(), 'retrained': False}
        write_json(out/'training.json', training)
        write_json(out/'manifest.json', {'outputs': {p.name: hashes(p)['sha256'] for p in out.iterdir()
                                                   if p.is_file() and p.name != 'INCOMPLETE'}})
        (out/'INCOMPLETE').unlink()
        print(f'Published {len(inputs.item_ids):,} movie vectors to {collection}', flush=True)
        return index.descriptor
    finally:
        index.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--collection', required=True)
    args = parser.parse_args()
    publish(args.inputs_dir, args.model_dir, args.output_dir, args.collection)


if __name__ == '__main__':
    main()
