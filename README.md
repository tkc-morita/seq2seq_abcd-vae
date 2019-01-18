# LSTM AUDIO AUTOENCODER

This is a PyTorch-implementation of an LSTM seq2seq autoencoder (AE).

## Dependencies

We tested the programs in the following environment:
- (mini)conda 4.5.12
- PyTorch 1.0

## Structure

- `./`
	- [`code/`](./code/): Code.
		- [`learning.py`](./code/learning.py): Execute learning.
		- [`model.py`](./code/model.py): Model module.
		- [`data_utils.py`](./code/data_utils.py): Utilities for data.
	- [`toy_data/`](./toy_data/): Toy data.
		- `annotation_20170806-080002_89.2-94.22.csv`
		- `20170806-080002_89.2-94.22.1ch.wav`

## How to use

### Learning

```bash
python code/learning.py /path/to/directory/containing/wavs /path/to/annotation.csv [options]
```

e.g. Train on the toy data for 20 epoches and save the results under `./results/toy`.
```bash
python code/learning.py data/toy/ data/toy/annotation_20170806-080002_89.2-94.22.csv -S results/toy -e 20
```

Options:
DESCRIPTION TO BE ADDED HERE.

### Encoding

UNDER CONSTRUCTION.

### Decoding

UNDER CONSTRUCTION.

## TODOs

- Check CUDA (10) compatibility.
- Implement post-learning encoder.
- Implement post-learning decoder.