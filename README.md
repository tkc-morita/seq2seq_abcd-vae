# Seq2Seq ABCD-VAE

This is a PyTorch-implementation of the sequence-to-sequence variational autoencoder that classifies audio segments (of variable length) into discrete categories based on the Attention-Based Categorical sampling with the Dirichlet prior (seq2seq ABCD-VAE).

## Dependencies

We tested the programs in the following environment:
- PyTorch 1.2

## Main Programs

- `./`
	- [`ABCD-VAE/`](./ABCD-VAE/): End-to-End clustering by the VAE with the Attention-Based Categorical sampling with the Dirichlet prior.
		- [`learning.py`](./ABCD-VAE/learning.py): Execute learning.
		- [`encode.py`](./ABCD-VAE/encode.py): Post-learning encoding.
	- [`plain/`](./plain/): Vanilla VAE with the Gaussian noise.
		- [`learning.py`](./plain/learning.py): Execute learning.
		- [`encode.py`](./plain/encode.py): Post-learning encoding.
	- [`toy_data/`](./toy_data/): Toy data.
		- `annotation_20170806-080002_89.2-94.22.csv`
		- `20170806-080002_89.2-94.22.1ch.wav`

## How to use

### Data

Two types of data are needed.
- .wav files including recordings. They must be located under some 
- a .csv file containing segmentation info etc. Each row corresponds to a single sound segment. Minimally required columns are:
  - `input_path`: Relative path from `/path/to/directory/containing/wavs` to the .wav file.
  - `onset`: Onset time in sec of the segment within the .wav file.
  - `offset`: Offset time in sec of the segment within the .wav file.
  - `data_type`: Either "train" (used for updating parameters) or "valid" (used for monitoring loss).
  - `speaker` (to use `--speaker_embed_dim` option of the programs): ID (str, int, etc) of the individual who produced the sound segment.

| input_path | onset   | offset  | speaker | ... |
| ---        | ---     | ---     | ---     | --- |
| 210101.wav |  0.2997 |  1.1135 | B01     | ... |
| 210503.wav |  1.4142 |  3.1415 | B01     | ... |
| 210503.wav | 21.7354 | 22.8521 | B01     | ... |
| ...        | ...     | ...     | ...     | ... |
| 201212.wav | 87.9836 | 89.6742 | B02     | ... |
| 170829.wav | 11.1212 | 12.3098 | B02     | ... |
| ...        | ...     | ...     | ...     | ... |

### Learning

```bash
python ABCD-VAE/learning.py /path/to/directory/containing/wavs /path/to/segmentation.csv -S /path/to/directory/to/save/results [options]
```

e.g. Train on the toy data for 20 epoches and save the results in `./results/toy/run-1`.
```bash
python ABCD-VAE/learning.py data/toy/ data/toy/annotation_20170806-080002_89.2-94.22.csv -S results/toy -j run-1 -e 20
```

Major options (values used in Morita et al, to appear, are in parentheses):
- `-S`: Path to the root directory under which results are saved.
- `-j`: Name of the directory created under the `-S` directory. Results are saved here.
- `-b`: Batch size. (512)
- `-e`: # of epochs to run. (20)
- `-p`: # of epochs before updating the learning rate when validation loss does not improve. (0)
- `-c`: Gradients are clipped if their norm exceeds this value. (1.0)
- `-d`: Device to use. Choose `cuda` for GPU learning.
- `-R`: Type of RNN architecture for encoder and decoder. (LSTM)
- `-N`: Denominator for rescaling the ln STFT amplitude. (11.0)
- `-K` (only for ABCD-VAE): Max # of classification categories assumed. (128)
- `-f`: Dimensionality of (pre-discretized) embeddings (i.e., codebook dimensionality of ABCD-VAE). (256) 
- `--encoder_rnn_hidden_size`: Dimensionality of hidden states in encoder RNN. (256)
- `--decoder_rnn_hidden_size`: Dimensionality of hidden states in decoder RNN. (256)
- `--mlp_hidden_size`: Dimensionality of hidden layers in MLP modules. (256)
- `--speaker_embed_dim`: Dimensionality of speaker embeddings. (256)
- `--pretrain_epochs` (only for ABCD-VAE): # of epochs during which ABCD-VAE feeds attention-weighted average of codebook vectors w/o Gumbel-Softmax sampling. (5)
- `--fft_frame_length`: Frame length of STFT in sec (0.008)
- `--fft_step_size`: Step size (or stride) of STFT in sec (0.004)

### Encoding

```bash
python ABCD-VAE/encode.py /path/to/training/checkpoint.pt /path/to/directory/containing/wavs /path/to/segmentation.csv value_of_-N_in_learning -S /path/to/class_probs.csv [options]
```

where `value_of_-N_in_learning` must be the float value used for the `-N` option of `learning.py`.

Major options:
- `-S`: Path to the csv file in which results are saved.
- `-b`: Batch size.
- `-d`: Device to use. Choose `cuda` for GPU learning.
- `--fft_frame_length`: Frame length of STFT in sec (0.008)
- `--fft_step_size`: Step size (or stride) of STFT in sec (0.004)

If you want pre-softmax classification logits, replace `ABCD-VAE/encode.py` with `ABCD-VAE/encode_logit.py`.

If you want pre-logit feature vectors (whose scaled dot-product with the categories' feature vectors), replace `ABCD-VAE/encode.py` with `ABCD-VAE/encode_features.py`.

## Citations

Please cite the following works when you refer to the ABCD-VAE.
- Morita, T., Koda, H., Okanoya, K., & Tachibana, R. O. (To appear) Measuring context dependency in birdsong using artificial neural networks. PLOS Computational Biology. ([Preprint available here](https://doi.org/10.1101/2020.05.09.083907))
- Morita, T. & Koda, H. (2020) Exploring TTS without T Using Biologically/Psychologically Motivated Neural Network Modules (ZeroSpeech 2020). *In* Proceedings of Interspeech 2020. 4856-4860. [DOI:10.21437/Interspeech.2020-3127](http://dx.doi.org/10.21437/Interspeech.2020-3127).



<!-- ## TODOs

- Check CUDA (10) compatibility.
- Implement post-learning decoder. -->