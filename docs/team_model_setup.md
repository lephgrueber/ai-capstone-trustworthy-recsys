# Download and run the trained model

This bundle provides the trained weights, all 50,986 movie embeddings, training configuration, manifests, movie/user mappings, and complete prepared model inputs. You do **not** need to rerun the data pipeline or train the model. You do need to import the saved embeddings into your own Weaviate instance once.

The complete prepared inputs are included because the current loader verifies their manifest and opens history and training arrays. This preserves the original hashes and known-user seen-item filtering. The raw MovieLens CSVs, held-out evaluation exports, and another teammate's Docker volume are not included.

## 1. Get the repository and binary bundle

Install Git LFS using the [official installation instructions](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage). With Git and Git LFS installed:

```powershell
git lfs install
git clone https://github.com/lephgrueber/ai-capstone-trustworthy-recsys.git
cd ai-capstone-trustworthy-recsys
git lfs pull --include="models/two_tower_80_v1.zip"
```

For an existing clone, pull the latest code and run the same `git lfs pull` command. If the repository is private, use your GitHub account with repository access.

Do not assume GitHub's **Download ZIP** contains the model: repository source archives can contain only the small LFS pointer, depending on repository settings. Use Git LFS to obtain the actual model bundle. See [GitHub's archive documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/managing-git-lfs-objects-in-archives-of-your-repository).

## 2. Install the Python environment and extract

From the repository root, using Python 3.11 or later:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[retrieval]"
.\.venv\Scripts\python.exe -m trustworthy_recsys.retrieval.bundle unpack --bundle models/two_tower_80_v1.zip --output-dir data/downloads/two_tower_80_v1
```

The extraction command checks the archive's SHA-256 against the adjacent `.sha256` file, then verifies each enclosed file. The output directory must be new; skip extraction if you already successfully unpacked this version. On macOS/Linux, use `.venv/bin/python` instead of `.\.venv\Scripts\python.exe`.

## 3. Start Weaviate and import the embeddings once

Start Docker Desktop with Linux containers enabled (or Docker Engine on Linux):

```powershell
docker compose up -d
.\.venv\Scripts\python.exe -m trustworthy_recsys.retrieval.publish --inputs-dir data/downloads/two_tower_80_v1/inputs --model-dir data/downloads/two_tower_80_v1/model --output-dir artifacts/retrieval/team_80_weaviate --collection MovieLensTeam80V1
```

Wait for Weaviate to be ready if the first connection fails during startup. This command copies the saved checkpoint and imports its vectors; it does not train. It verifies every stored vector and movie ID. Publication refuses an existing output directory or collection name. Each teammate can use the same name in their own local database; a shared database needs a unique collection name per publication.

## 4. Request recommendations

```powershell
.\.venv\Scripts\python.exe -m trustworthy_recsys.retrieval.predict --inputs-dir data/downloads/two_tower_80_v1/inputs --model-dir artifacts/retrieval/team_80_weaviate --user-id demo-user --history-items 1 260 1196 --k 10
```

The output contains original MovieLens IDs. Supply previously liked movies in `--history-items`; omit history for popularity fallback. For a known training user ID, all that user's previously rated training movies are excluded too. The initial load audits remote embeddings, so startup is slower than one recommendation query.

On later sessions, start Docker and run the prediction command. You do not need to extract or publish again. The database persists in a named Docker volume; removing that volume requires republishing.

## Included files and provenance

| Location inside the archive | Contents |
|---|---|
| `model/` | `model.pt`, `item_embeddings.npy`, `training.json`, `manifest.json`, `weaviate.json`, training pair indices and validation user IDs |
| `inputs/` | Movie/user mappings, features, all-rated and positive histories, target/context arrays, statistics in `metadata.json`, schema, Data Card and manifest |
| `bundle_manifest.json` | SHA-256 hashes for every payload file and the original input/model manifest hashes |
| `MOVIELENS_README.txt` | Original source description, redistribution/use conditions and citation requirements |
| `SOURCE_DATA_CARD.md`, `MODEL_CARD.md` | Source-data and retrieval-model documentation |
| `README.md` | These setup instructions |

Embedding row `i` maps to movie `item_ids.json[i]`. The source artifact manifests are included unchanged. Historical provenance fields may contain the original machine's source-data path; prediction and publication do not read that path. Evaluation/training commands require the source split exports and are not the purpose of this download.

The packaged `weaviate.json` describes the original collection. The publication command creates a descriptor for your own collection, which is why prediction uses `artifacts/retrieval/team_80_weaviate` afterward.

MovieLens data and transformations remain subject to the conditions in `MOVIELENS_README.txt`, including attribution and restrictions on commercial/revenue-bearing use. The repository's software license does not replace those dataset terms. This bundle is intended for the team's research workflow.
