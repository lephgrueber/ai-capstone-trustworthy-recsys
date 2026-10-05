# Trained model download

`two_tower_80_v1.zip` contains the trained two-tower model and the complete prepared inputs required to load it. The bundle is about **124 MiB compressed** and is stored with Git LFS. Its SHA-256 is in `two_tower_80_v1.zip.sha256`.

Use [the teammate setup guide](../docs/team_model_setup.md) to download, verify, extract, import into your own Weaviate instance, and request recommendations. No data preparation or retraining is needed.

```powershell
git lfs pull --include="models/two_tower_80_v1.zip"
```

The archive contains trained weights, 50,986 movie vectors of dimension 32, `training.json`, original artifact manifests, movie/user mappings, features and histories, a per-file bundle manifest, setup instructions, Data/Model Cards, and the original MovieLens README with dataset terms and citation requirements.

This is the 80/10/10 model already evaluated in the repository. A teammate creates their own Weaviate collection from the saved embeddings; another machine's Docker database is not part of the download. The raw dataset and held-out evaluation exports are not included.

For a future model version, create a new versioned ZIP and checksum instead of overwriting this release. The packaging command is `python -m trustworthy_recsys.retrieval.bundle pack --help`.
