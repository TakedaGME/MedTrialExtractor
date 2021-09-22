#!/usr/bin/env python
# coding: utf-8

import json
import sys
import os
from glob import glob
import pandas as pd
import pickle
from bisect import bisect_right, bisect_left
from argparse import ArgumentParser
from tqdm import tqdm


def parse_annotation_file(ann_path):
    annotations = []
    errors = []
    with open(ann_path) as ann_file:
        ann_lines = ann_file.readlines()

        for l in ann_lines:

            # Correct parsings
            try:
                line_segments = [e.strip() for e in l.split('\t')]
                a_name = line_segments[0]

                if a_name.startswith('T'):
                    a_type, a_start_idx, a_stop_idx = line_segments[1].split(
                        ' ')
                    a_txt = line_segments[2]
                    ann = {
                        'name': a_name,
                        'type': 'ent',
                        'label': a_type,
                        'start_idx': int(a_start_idx),
                        'stop_idx': int(a_stop_idx),
                        'text': a_txt
                    }
                elif a_name.startswith('R'):
                    a_type, arg1, arg2 = line_segments[1].split(' ')
                    ann = {
                        'name': a_name,
                        'type': 'rel',
                        'label': a_type,
                        'arg1': arg1.split(':')[1],
                        'arg2': arg2.split(':')[1]
                    }

                annotations.append(ann)
            except:
                # Problematic parsings (e.g. multi-span ann)
                errors.append(l)

    annotations = pd.DataFrame(annotations)

    return annotations, errors


def parse_text_file(txt_path):

    paragraphs = []
    all_text = ''

    with open(txt_path, 'r') as txt_file:
        txt_lines = txt_file.readlines()

        offset = 0
        for j, l in enumerate(txt_lines):

            header_split_idx = l.find(':')
            header = l[:header_split_idx]
            par_offset = header_split_idx + 2

            txt_line = {
                'idx': j,
                'par_offset': offset,
                'text_offset': par_offset,
                'header': header,
                'text': l
            }

            offset += len(l)
            all_text += l

            paragraphs.append(txt_line)

    paragraphs = pd.DataFrame(paragraphs)

    return paragraphs, all_text


# In[53]:


def parse_document_annotations(ann_file_path, txt_file_path):
    basename = os.path.basename(ann_file_path)
    basename = basename[:basename.rfind('.')]

    anns, errs = parse_annotation_file(ann_file_path)
    paragraphs, all_text = parse_text_file(txt_file_path)

    paragraphs['entities'] = None
    anns['par_idx'] = None

    anns['document'] = basename
    paragraphs['document'] = basename

    offsets = list(paragraphs['par_offset'].values)
    last_len = len(paragraphs.iloc[len(paragraphs) - 1]['text'])
    offsets.append(offsets[-1] + last_len)

    # Add entities to paragraph dataframe
    for j, ann_row in anns.loc[anns['type'] == 'ent'].iterrows():
        start = int(ann_row['start_idx'])
        stop = int(ann_row['stop_idx'])

        par_idx = bisect_left(offsets, stop) - 1
        anns.at[j, 'par_idx'] = par_idx

        pstart, pstop = offsets[par_idx: par_idx + 2]

        str_start = start - pstart
        str_stop = stop - pstart
        par_text = paragraphs.iloc[par_idx]['text'][str_start:str_stop]
        ann_txt = ann_row['text']

        par_ents = paragraphs.at[par_idx, 'entities']
        if par_ents is None:
            paragraphs.at[par_idx, 'entities'] = list()
            par_ents = paragraphs.at[par_idx, 'entities']

        ent_entry = dict(ann_row[['name', 'label']])
        ent_entry['idxs'] = (str_start, str_stop)
        par_ents.append(ent_entry)

        assert(par_text == ann_txt)

    return paragraphs, anns


def parse_driver(input_dir, output_dir, dataset_name='dataset'):

    ann_template = os.path.join(os.path.realpath(input_dir), '**', '*.ann')
    annotation_file_paths = glob(ann_template, recursive=True)

    ann_dfs = []
    par_dfs = []

    for ann_file_path in tqdm(annotation_file_paths):
        txt_file_path = ann_file_path.replace('.ann', '.txt')
        par_df, anns_df = parse_document_annotations(
            ann_file_path, txt_file_path)
        ann_dfs.append(anns_df)
        par_dfs.append(par_df)

    annotations = pd.concat(ann_dfs).reset_index(drop=True)
    paragraphs = pd.concat(par_dfs).reset_index(drop=True)

    output_dict = {'paragraphs': paragraphs,
                   'annotations': annotations}

    output_path = os.path.join(output_dir, f'{dataset_name}.pkl')
    with open(output_path, 'wb') as output_file:
        pickle.dump(output_dict, output_file)

    return annotations, paragraphs


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('input_dir', help='Path to input directory')
    parser.add_argument('output_dir', help='Path to output directory')
    parser.add_argument('--dataset_name', help='name of dataset')

    # Parse arguments
    try:
        args = parser.parse_args()
        # Execute and save extractions
        kwargs = vars(args)
        parse_driver(**kwargs)
    except:
        parser.print_help()
