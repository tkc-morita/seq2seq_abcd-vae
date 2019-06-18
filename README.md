# RNN Audio Variational Autoencoder

This is a PyTorch-implementation of an RNN seq2seq variational autoencoder (RNN-VAE).

## Dependencies

We tested the programs in the following environment:
- (mini)conda 4.5.12
- PyTorch 1.1

## Main Programs

- `./`
	- [`plain/`](./plain/): Code for autoencoding into a continuous feature space.
		- [`learning.py`](./plain/learning.py): Execute learning.
		- [`encode.py`](./plain/encode.py): Post-learning encoding.
		- [`decode.py`](./plain/decode.py): Post-learning decoding.
	- [`clustering/`](./clustering/): Code for network-internal clustering by GMM with Dirichlet prior.
		- [`learning.py`](./clustering/learning.py): Execute learning.
		- [`encode.py`](./clustering/encode.py): Post-learning encoding.
		- [`decode.py`](./clustering/decode.py): Post-learning decoding.
	- [`toy_data/`](./toy_data/): Toy data.
		- `annotation_20170806-080002_89.2-94.22.csv`
		- `20170806-080002_89.2-94.22.1ch.wav`

## How to use

### Learning

```bash
python plain/learning.py /path/to/directory/containing/wavs /path/to/annotation.csv [options]
```

e.g. Train on the toy data for 20 epoches and save the results under `./results/toy`.
```bash
python plain/learning.py data/toy/ data/toy/annotation_20170806-080002_89.2-94.22.csv -S results/toy -e 20
```

Options:
DESCRIPTION TO BE ADDED HERE.

### Encoding

```bash
python plain/encode.py /path/to/directory/containing/learning_results /path/to/directory/containing/wavs /path/to/annotation.csv [options]
```

Options:
DESCRIPTION TO BE ADDED HERE.

### Decoding

UNDER CONSTRUCTION.

## TODOs

- Check CUDA (10) compatibility.
- Implement post-learning decoder.