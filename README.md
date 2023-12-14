# ecog2vec

Learning latent representations on ECoG data

### 1: How to run the experiments and 2: code I borrowed/modified vs. code of my own

```
git clone --recurse-submodules git@github.com:b4yuan/ecog2vec.git
pip install -r requirements.txt

cd packages
pip install -e ecog2vec
pip install -e vec2txt

pip install -e fairseq

pip install -e utils_jgm
pip install -e machine_learning
pip install -e ecog2txt
```

The `ecog2vec` and `vec2txt` packages and all their contents were all **written by me** to assist in the experimental process. The `ecog2txt` package essentially provides a class that functions as a dataloader for inputs to the `fairseq` package. The `fairseq` package was taken directly from Facebook AI (https://github.com/facebookresearch/fairseq) and modified to change the expected single-channel input to handle multi-channel inputs. Maybe 10 lines of code from `fairseq` was changed by me. Some functions scattered throughout the project (that are cited) were borrowed from https://github.com/jgmakin/ecog2txt. 

`/notebooks` contains notebooks that was used to run the experiments.

- `/notebooks/wav2vec.ipynb`: very initial testing with `facebookresearch/fairseq`; here I'm taking `wav2vec` out of the box and experimentingly purely with functionality. Essentially all the code is taken from Facebook AI.
- `/notebooks/ecog2vec.ipynb`: here, I'm loading in the ECoG recordings using the `ecog2vec` package. As the cells prior to the shell commands use the `DataGenerator` class from `ecog2vec`, these were written by me. After those cells, the shell commands and the feature extracting borrowed heavily from scripts provided by Facebook AI.
- Other misc. notebooks: they include some code, written by myself and not written by myself. These are not important; I have consolidated all of the experiments into the previously mentioned notebooks, but these are included for documentation. They also may serve useful to me in the future, but they are far more messily organized. To be perfectly explicit: these notebooks are essentially a melting pot of work from myself and sources for my project. These notebooks do not need to be included for my project to be replicated.

Two other scripts are also very useful to decode from latent representations to text:
- `/packages/vec2txt/train_on_preprocessed_hg.py`: The nwbfiles that contain the ECoG recordings also include data that has been fed through the preprocessing pipeline of `jgmakin/ecog2txt`. This preprocessed data was treated as the ground truth; from there, I wanted to see how well the latent representations would perform. **Functions that borrow/modify code that isn't mine is explicitly cited**; any code that isn't cited is to be assumed as mine. This script takes the preprocessed data, and runs it through the RNN described in `/packages/vec2txt/vec2txt/model.py` to decode it to text.
- `/packages/vec2txt/train_on_latent_rep.py`: This essentially does the same thing as the above, but on the latent representations from the output of `ecog2vec` and the modified `wav2vec`. Again, **code that isn't explicitly cited is to be assumed as mine**. This script runs the latent representations through the RNN described in `/packages/vec2txt/vec2txt/model.py` to decode it to text.

Other folders:
- `/manifest`, `/model`, `/runs`: training manifest for the modified `wav2vec`, model checkpoints for the modified `wav2vec`, and a runs folder for each training run of the modified `wav2vec` in order to be viewed in tensorboard.

### 3: Dataset used

This project used ECoG recordings from two subjects (EFC400 and EFC401) from Professor Makin's paper https://www.nature.com/articles/s41593-020-0608-8. The dataset was provided by Professor Makin, but they are in the public domain. 