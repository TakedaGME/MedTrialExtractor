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

Task | Precision | Recall | F1 |
| :---: | :---: | :---: | :---: |
Product Extraction | 84.62 | 69.37 | 76.24 |
Role Extraction | 80.12 | 77.25 | 78.66 |

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

