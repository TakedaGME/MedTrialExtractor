# Chemical Reaction Extraction from Scientific Literature

This repository contains code/data for extracting chemical reactions from scientific literature.

## Installation

### Pre-requirements

1. pytorch (>=1.5.0)
2. transformers (tested on v3.0.2)
3. tqdm (>=4.36.0)
4. numpy (>=1.18.0)
5. seqeval

### Install from source
1. `git clone https://github.com/jiangfeng1124/medtrialextractor`
2. `cd medtrialextractor`
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

model_dir="models" # directory saving both ner and role models
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

We provide scripts to train new models (neruct/role extraction) using your own data. We also plan to release our training data in the near future.

#### Data format

Your training data should contain texts (sequences of tokens) and known target labels.
We follow conventional BIO-tagging scheme, where `B-{type}` indicates the Beginning of a specific entity type (e.g., ner, Reactants, Solvent), and `I-{type}` means the Inside of an entity.

##### Product Extraction

The train/dev/test files have the same CoNLL-style format:
```csv
#	passage=10.1021/ja00020a078-5	sentence=1
Reaction	O
of	O
diphenylacetylene	O
with	O
complex	O
19A	O
led	O
to	O
only	O
cycloheptadienone	B-arm_description
23A	B-arm_description
in	O
30	O
%	O
yield	O
```

It is assumed that the tokens are in the first column, and the targets are in the second column.
The comment line (optional) can contain any meta information of the current text sequence, such as the DOI of a paper.

##### Reaction Role Extraction

Data files for role extraction can have multiple label columns, each corresponding to one product. For example:
```csv
#	passage=10.1021/ja00020a078-5	segment=1
Reaction	O	O	O
of	O	O	O
diphenylacetylene	B-Reactants	B-Reactants	B-Reactants
with	O	O	O
complex	O	O	O
19A	B-Reactants	B-Reactants	O
led	O	O	O
to	O	O	O
only	O	O	O
cycloheptadienone	B-arm_description	O	O
23A	O	B-arm_description	O
in	O	O	O
30	B-Yield	B-Yield	O
%	I-Yield	I-Yield	O
yield	O	O	O
;	O	O	O
with	O	O	O
(phenylcyclopropy1)-	O	O	O
carbene	O	O	O
complex	O	O	O
19B	O	O	B-Reactants
,	O	O	O
cycloheptadienone	O	O	O
25	O	O	B-arm_description
was	O	O	O
produced	O	O	O
in	O	O	O
53	O	O	B-Yield
%	O	O	I-Yield
yield	O	O	O
```

The tokens are in the first column, and the target labels are in the remaining columns.

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
Please create an issue or email to [jiang_guo@csail.mit.edu](mailto:jiang_guo@csail.mit.edu) should you have any questions, comments or suggestions.

