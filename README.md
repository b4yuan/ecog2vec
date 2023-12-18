# ecog2vec

Learning latent representations on ECoG data

### 1: Setup

```
git clone --recurse-submodules git@github.com:b4yuan/ecog2vec.git
pip install -r requirements.txt

cd packages
pip install -e ecog2vec
<!-- pip install -e vec2txt -->

pip install -e fairseq

pip install -e utils_jgm
pip install -e machine_learning
pip install -e ecog2txt
```

## Suggested file structure

```
ecog2vec
|-- manifest
|-- model
|-- notebooks
|-- packages
|   |-- ecog2txt
|   |-- ecog2vec
|   |-- fairseq
|   |-- machine_learning
|   |-- utils_jgm
|-- runs
|-- wav2vec_inputs
|-- wav2vec outputs
|-- wav2vec_tfrecords
```

Three main jupyter notebooks in `notebooks/` are necessary for running this pipeline.

1. `ecog2vec`: This notebook trains a `wav2vec` model on ECoG and extracts features.
2. `vec2txt`: This notebook runs `ecog2txt` to decode from the `wav2vec` features.
3. `original_tf_records.ipynb`: This notebook creates tf_records as inputs to `ecog2txt` to measure a baseline performance.

Currently looking to incoporate `wav2vec 2.0`. Coming soon.