import os
from setuptools import setup, find_packages

__version__ = None

src_dir = os.path.abspath(os.path.dirname(__file__))
version_file = os.path.join(src_dir, 'medtrialextractor', '_version.py')

with open(version_file, encoding='utf-8') as fd:
    exec(fd.read())

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='medtrialextractor',
    author="Jiang Guo, Santiago Ibanez, Hanyu Gao",
    author_email="jiang_guo@csail.mit.edu",
    description='Chemical Reaction Extraction from Literature',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='Pre-trainin://github.com/jiangfeng1124/ChemRxnExtractor',
    version=__version__,
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy==1.20.2',
        'seqeval==1.2.2',
        'torch==1.8.1',
        'tqdm==4.60.0',
        'transformers==3.0.2',
        'ya.dotdict',
        'beautifulsoup4',
        'spacy',
        'pandas'
    ],
    keywords=[
        'information extraction',
        'reaction extraction',
        'natural language processing',
        'pre-training'
    ]
)
