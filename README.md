# Full Publication Data Extraction for Systematic Literature Review

This repository contains code/data for extraction of entities and relationships from scientific literature for the purposes of a systematic review.  The code is based on a previous project extracting chemical reactions from scientific literature.

## Installation

### Pre-requirements

1. pytorch (>=1.5.0)
2. transformers (tested on v3.0.2)
3. tqdm (>=4.36.0)
4. numpy (>=1.18.0)
5. seqeval

### Install from source
1. `git clone https://github.com/TakedaGME/MedTrialExtractor`
2. `cd ChemRxnExtractor`
3. `pip install -r requirements.txt`
4. `pip install -e .`

### Download Trained Models
Download the trained models: [cre_models_v0.1.tgz](https://drive.google.com/file/d/1HeP2NlSAdqNzlTqmHCrwmoUNiw9JWdaf/view?usp=sharing), and extract to the current directory:
```bash
tar zxvf cre_models_v0.1.tgz
```

## Usage

Using RxnExtractor in your code:
```python
from medtrialextractor import RxnExtractor

model_dir="models" # directory saving both prod and role models
rxn_extractor = RxnExtractor(model_dir)

# test_file contains texts line by line
with open(test_file, "r") as f:
    sents = f.read().splitlines()
rxns = rxn_extractor.get_reactions(sents)
```

`model_dir` points to the directory of the trained models (e.g., `cre_models_v0.1`).
`test_file` has an independent paragraph/sentence each line (e.g., `tests/sample_data/raw.txt`). See `pipeline.py` for more details.
GPU is used as the default device, please ensure that you have at least >5G allocatable GPU memory.

**Preprocessing:** We recommend using the [ChemDataExtractor](http://chemdataextractor.org/) toolkit for the preprocessing of chemical documents in PDF format, such as PDF parsing, sentence segmentation, and tokenization.

## Train and Evaluation

### Pre-training: ChemBERT

Our model is greatly benefited from a domain-adaptively pre-trained model named **ChemBERT**.
To train a new model on your own datasets, download [ChemBERT v3.0](https://drive.google.com/file/d/1UMYYD9P8fJgs61FJc06sRbbdDxOYPbMu/view?usp=sharing), and extract to a local directory.

### Fine-tuning

We provide scripts to train new models (product/role extraction) using your own data. We also plan to release our training data in the near future.

#### Data format

Your training data should contain texts (sequences of tokens) and known target labels.
We follow conventional BIO-tagging scheme, where `B-{type}` indicates the Beginning of a specific entity type (e.g., Prod, Reactants, Solvent), and `I-{type}` means the Inside of an entity.

##### Extraction

The train/dev/test files have the same CoNLL-style format:
```csv
The\O
main\O
objective\O
of\O
the\O
18\O
-\O
month\O
,\O
randomised\O
,\O
active\O
-\O
controlled\O
ATTRACT\O
study\O
was\O
to\O
assess\O
the\O
effects\O
of\O
migalastat\B-drug
on\O
renal\O
function\O
in\O
patients\O
with\O
Fabry\B-disease
disease\I-disease
.\O’

The tokens are in the first column, and the target labels are in the second columns.

#### Run
To train or evaluate a product extraction model, run:
```
python train.py <task> <config_path>|<options>
```
where `<task>` is either "prod" or "role" depending on the task of interest, `<config_path>` is a json file containing required hyper-parameters such as the paths to the model and the data; `<options>` are instead explicitly-specified hyper-parameters.

For example:
```
python train.py prod configs/prod_train.json
```

Configure `configs/prod_train.json` to turn on/off the train/eval modes.

## Performance

Performance of the provided trained models on our test set (`tests/sample_data/<task>/test.txt`):

Entity recognition performance across machine learning models
Model	Relaxed	Strict
Precision, %	Recall, %	F1 score, %	Precision, %	Recall, %	F1 score, %
SLR 1
BiLSTM+linear	68.1	58.8	63.0	45.5	39.3	42.1
BiLSTM+CRF	74.9	53.4	62.2	53.0	37.8	44.0
BERT+linear	67.3	66.8	67.0	46.3	45.7	46.0
BERT+CRF	73.5	64.5	68.7	52.3	45.8	48.8
Pretrained BERT+linear	68.1	71.7	69.8	47.5	49.8	48.6
Pretrained BERT+CRF	74.0	71.9	72.8	53.3	51.7	52.4
SLR 2
BiLSTM+linear	69.2	58.0	63.1	46.9	44.6	45.7
BiLSTM+CRF	73.3	56.3	63.4	55.3	42.2	47.7
BERT+linear    59.2	60.8	59.1	43.7	44.6	43.4
BERT+CRF	65.6	58.3	60.7	50.3	44.5	46.4
Pretrained BERT+linear	62.5	66.6	63.7	46.9	49.8	47.8
Pretrained BERT+CRF	69.7	70.5	69.5	55.8	56.0	55.4

BERT, bidirectional encoder representations from transformers; BiLSTM, bidirectional long-short-term memory; CRF, conditional random field; SLR, systematic literature review.

## Predict

To generate predictions for unlabeled inputs (see `tests/sample_data/<task>/inputs.txt` for the format of unlabeled inputs), run:
```
python predict.py <task> <config_json>
```

For example:
```
python predict.py prod configs/prod_predict.json
```

## Contact
Please create an issue or email to [antonia.panayi@takeda.com](mailto:antonia.panayi@takeda.com) should you have any questions, comments or suggestions.

