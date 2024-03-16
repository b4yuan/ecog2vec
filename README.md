# ecog2vec

Learning latent representations on ECoG.

### 1: Setup

Recommended: within a conda env

```
git clone --recurse-submodules git@github.com:b4yuan/ecog2vec.git
pip install -r requirements.txt

cd packages
pip install -e ecog2vec
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

## Train a model

Three main jupyter notebooks in `notebooks/`:

1. `ecog2vec`: This notebook trains a `wav2vec` model on ECoG and extracts features.
2. `vec2txt`: This notebook runs `ecog2txt` to decode from the `wav2vec` features.
3. `_original_tf_records.ipynb`: This notebook creates 'original' tf_records as inputs to `ecog2txt` to measure a baseline performance.

To allow w2v to accept inputs with multiple channels--two files need changed in the `fairseq` package:

1. https://github.com/b4yuan/fairseq/blob/9660fea38cfcdbec67e0e3aba8d7907023a36aa2/fairseq/data/audio/raw_audio_dataset.py#L138
2. https://github.com/b4yuan/fairseq/blob/9660fea38cfcdbec67e0e3aba8d7907023a36aa2/fairseq/models/wav2vec/wav2vec.py#L391

Set at 256 by default. Change to # of electrodes for the patient, less the bad electrodes.
