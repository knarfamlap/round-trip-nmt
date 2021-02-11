# wikimedia-2020-01-17.zip

* dataset: wikimedia
* model: transformer-align
* pre-processing: normalization + SentencePiece
* download: [wikimedia-2020-01-17.zip](https://object.pouta.csc.fi/OPUS-MT-models/bcl-en/wikimedia-2020-01-17.zip)
* test set translations: [wikimedia-2020-01-17.test.txt](https://object.pouta.csc.fi/OPUS-MT-models/bcl-en/wikimedia-2020-01-17.test.txt)
* test set scores: [wikimedia-2020-01-17.eval.txt](https://object.pouta.csc.fi/OPUS-MT-models/bcl-en/wikimedia-2020-01-17.eval.txt)

## Benchmarks

| testset               | BLEU  | chr-F |
|-----------------------|-------|-------|
| JW300.bcl.en 	| 56.8 	| 0.705 |

# opus-2020-01-20.zip

* dataset: opus
* model: transformer-align
* pre-processing: normalization + SentencePiece
* download: [opus-2020-01-20.zip](https://object.pouta.csc.fi/OPUS-MT-models/bcl-en/opus-2020-01-20.zip)
* test set translations: [opus-2020-01-20.test.txt](https://object.pouta.csc.fi/OPUS-MT-models/bcl-en/opus-2020-01-20.test.txt)
* test set scores: [opus-2020-01-20.eval.txt](https://object.pouta.csc.fi/OPUS-MT-models/bcl-en/opus-2020-01-20.eval.txt)

## Benchmarks

| testset               | BLEU  | chr-F |
|-----------------------|-------|-------|
| JW300.bcl.en 	| 56.8 	| 0.705 |

# opus-2020-02-11.zip

* dataset: opus
* model: transformer-align
* pre-processing: normalization + SentencePiece
* download: [opus-2020-02-11.zip](https://object.pouta.csc.fi/OPUS-MT-models/bcl-en/opus-2020-02-11.zip)
* test set translations: [opus-2020-02-11.test.txt](https://object.pouta.csc.fi/OPUS-MT-models/bcl-en/opus-2020-02-11.test.txt)
* test set scores: [opus-2020-02-11.eval.txt](https://object.pouta.csc.fi/OPUS-MT-models/bcl-en/opus-2020-02-11.eval.txt)

## Benchmarks

| testset               | BLEU  | chr-F |
|-----------------------|-------|-------|
| JW300.bcl.en 	| 56.1 	| 0.697 |

# opus+bt-2020-05-23.zip

* dataset: opus+bt
* model: transformer-align
* source language(s): bcl
* target language(s): en
* model: transformer-align
* pre-processing: normalization + SentencePiece (spm4k,spm4k)
* download: [opus+bt-2020-05-23.zip](https://object.pouta.csc.fi/OPUS-MT-models/bcl-en/opus+bt-2020-05-23.zip)
* test set translations: [opus+bt-2020-05-23.test.txt](https://object.pouta.csc.fi/OPUS-MT-models/bcl-en/opus+bt-2020-05-23.test.txt)
* test set scores: [opus+bt-2020-05-23.eval.txt](https://object.pouta.csc.fi/OPUS-MT-models/bcl-en/opus+bt-2020-05-23.eval.txt)

## Training data:  opus+bt

* bcl-en: wikimedia (1106) 
* bcl-en: total size = 1106
* unused dev/test data is added to training data
* total size (opus+bt): 458304


## Benchmarks

| testset               | BLEU  | chr-F |
|-----------------------|-------|-------|
| JW300.bcl.en 	| 57.6 	| 0.712 |

