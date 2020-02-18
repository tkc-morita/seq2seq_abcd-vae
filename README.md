# Seq2Seq Audio Variational Autoencoder

This is a PyTorch-implementation of an RNN sequence-to-sequence variational autoencoder (seq2seq-VAE).

## Dependencies

We tested the programs in the following environment:
- PyTorch 1.2

## Main Programs

- `./`
	- [`plain/`](./plain/): Vanilla VAE with the Gaussian noise.
		- [`learning.py`](./plain/learning.py): Execute learning.
		- [`encode.py`](./plain/encode.py): Post-learning encoding.
		- [`decode.py`](./plain/decode.py): Post-learning decoding.
	- [`ABCD-VAE/`](./ABCD-VAE/): End-to-End clustering by the VAE with the Attention-Based Categorical sampling with the Dirichlet prior.
		- [`learning.py`](./ABCD-VAE/learning.py): Execute learning.
		- [`encode.py`](./ABCD-VAE/encode.py): Post-learning encoding.
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

<!-- ## TODOs

- Check CUDA (10) compatibility.
- Implement post-learning decoder. -->