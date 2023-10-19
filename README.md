# ecog2vec

Clone and install [bayuan/fairseq](https://github.com/b4yuan/fairseq):

```
git clone git@github.com:b4yuan/ecog2vec.git
git clone git@github.com:b4yuan/fairseq.git
pip install --editable fairseq
```
`ecog2vec/notebooks/ecog2vec.ipynb` trains a wav2vec model on 256-channel inputs of ECoG data via unsupervised methods to learn some underlying representative vector $c$.


### Misc notebooks

`ecog2vec/notebooks/wav2vec_out_out_the_box.ipynb` creates random train/valid datasets and trains a wav2vec model as `facebookresearch/fairseq` would, with a single channel input. 