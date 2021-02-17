'''
Create BIO annotations from struct file
'''

import os
import json
import codecs

# NER


def struct_to_bio(args):
    struct_ann_path = args.struct_ann_path
    output_dir = args.output_dir
    oversampling_rate = args.oversampling_rate
    separate_docs = args.separate_docs

    bio_dict = struct_to_bio_dict(struct_ann_path, oversampling_rate)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if separate_docs:
        for doc_id in bio_dict:
            bio_file_path = os.path.join(output_dir, doc_id + '.bio')
            with codecs.open(bio_file_path, 'wb', encoding='utf-8', errors='replace') as bio_file:
                bio_file.write(bio_dict[doc_id])
    else:
        output_file_path = os.path.join(output_dir, 'paragraphs.bio')
        txt = '\n\n'.join(bio_dict.values())
        with codecs.open(output_file_path, 'wb', encoding='utf-8', errors='replace') as bio_file:
            bio_file.write(txt)


def struct_to_bio_dict(struct_ann_path, oversampling_rate, use_tags=True):

    # Read Struct
    struct = None
    with codecs.open(struct_ann_path, 'rb', encoding='utf-8', errors='replace') as struct_file:
        struct = json.load(struct_file)

    bio_dict = {}

    doc_ids = []
    if use_tags:
        doc_ids = struct['annotated_docs']
    else:
        doc_ids = struct['documents'].keys()

    for doc_id in doc_ids:
        bio_dict[doc_id] = create_doc_bio_annotations(
            struct['documents'][doc_id], oversampling_rate, use_tags=use_tags)

    return bio_dict


ner_tags = ['authors', 'study_type', 'arm_efficacy_metric',
            'arm_efficacy_results', 'arm_dosage', 'arm_description']


def create_doc_bio_annotations(doc_struct, oversampling_rate=1, use_tags=True, filters={}):
    bio_pars = []

    if use_tags:
        for par in doc_struct['paragraphs']:

            if not par['annotated']:
                continue

            toks = par['text'].split(' ')
            labs = ['O'] * len(toks)
            spans = par['annotations_spans']

            is_empty = True
            for k in ner_tags:
                if k in spans:
                    for i, j in spans[k]:
                        is_empty = False
                        labs[i] = f'B-{k}'
                        for l_idx in range(i + 1, j):
                            labs[l_idx] = f'I-{k}'

            lines = ['\t'.join(e) for e in zip(toks, labs)]
            par_txt = '\n'.join(lines)

            reps = 1 if is_empty else oversampling_rate
            for _ in range(reps):
                bio_pars.append(par_txt)
    else:
        for par in doc_struct['paragraphs']:

            toks = par['text'].split(' ')
            par_txt = '\n'.join(toks)
            bio_pars.append(par_txt)

    return '\n\n'.join(bio_pars)


# RD
default_trigger = 'arm_description'
non_numbered_foi = ['arm_efficacy_metric', 'arm_efficacy_results']
numbered_foi = ['arm_dosage', 'arm_description']
foi = non_numbered_foi + numbered_foi


def struct_to_bio_rd(args):

    struct_ann_path = args.struct_ann_path
    output_dir = args.output_dir
    separate_docs = args.separate_docs

    bio_dict = struct_to_bio_dict_rd(struct_ann_path, output_dir)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if separate_docs:
        for doc_id in bio_dict:
            bio_file_path = os.path.join(output_dir, doc_id + '.bio')
            with codecs.open(bio_file_path, 'wb', encoding='utf-8', errors='replace') as bio_file:
                bio_file.write(bio_dict[doc_id])
    else:
        output_file_path = os.path.join(output_dir, 'paragraphs.bio')
        txt = '\n\n'.join(bio_dict.values())
        with codecs.open(output_file_path, 'wb', encoding='utf-8', errors='replace') as bio_file:
            bio_file.write(txt)


def struct_to_bio_dict_rd(struct_ann_path, is_pred=False):

    # Read Struct
    struct = None
    with codecs.open(struct_ann_path, 'rb', encoding='utf-8', errors='replace') as struct_file:
        struct = json.load(struct_file)

    bio_dict = {}

    print('Creating empty docs...')

    doc_ids = []
    if not is_pred:
        doc_ids = struct['annotated_docs']
    else:
        doc_ids = struct['documents'].keys()

    for doc_id in doc_ids:
        bio_dict[doc_id] = create_doc_bio_annotations_rd(
            struct['documents'][doc_id], is_pred=is_pred)


    return bio_dict


def create_doc_bio_annotations_rd(doc_struct, is_pred=False, filters={}):
    par_dicts = []

    if not is_pred:
        for par in doc_struct['paragraphs']:

            if not par['annotated']:
                continue

            spans = par['annotations_spans']
            arms = par['annotations_span_arms']
            par_annotations = dict()

            for k in spans:
                if k in arms:
                    for a_count, d_num in arms[k]:
                        if a_count != '*' and d_num != '*':
                            toks = par['text'].split(' ')
                            par_annotations[(a_count, d_num)] = {
                                'toks': toks,
                                'labs': ['O'] * len(toks)
                            }

            for k in foi:
                if k in spans:
                    for j, (start, stop) in enumerate(spans[k]):

                        arm_key = arms[k][j]
                        a_count, d_num = arm_key

                        for ac, dn in par_annotations:
                            if (ac == a_count or ac == '*') and (dn == d_num or dn == '*'):
                                # print((ac, dn), arm_key)
                                if k != default_trigger:
                                    par_annotations[(
                                        ac, dn)]['labs'][start] = f'B-{k}'
                                    for l_idx in range(start + 1, stop):
                                        par_annotations[(
                                            ac, dn)]['labs'][l_idx] = f'I-{k}'
                                else:
                                    if 'desc_range' not in par_annotations[(ac, dn)]:
                                        par_annotations[(
                                            ac, dn)]['desc_range'] = []
                                    par_annotations[(ac, dn)]['desc_range'].append(
                                        (start, stop))
            par_dicts.extend(list(par_annotations.values()))

        bio_pars = []
        for ann in par_dicts:
            if 'desc_range' in ann:
                toks = ann['toks']
                labs = ann['labs']
                lines = ['\t'.join(e) for e in zip(toks, labs)]

                for i, j in ann['desc_range']:
                    desc_lines = list(lines)
                    # Delineate trigger entity
                    desc_lines.insert(j, '[P2]\tO')
                    desc_lines.insert(i, '[P1]\tO')
                    desc_txt = '\n'.join(desc_lines)
                    bio_pars.append('\n'.join(desc_lines))

        return '\n\n'.join(bio_pars)

    else:  # Using during prediction

        bio_pars = []

        for par in doc_struct['paragraphs']:

            if ('predictions' not in par) or\
                ('ner' not in par['predictions']) or\
                (default_trigger not in par['predictions']['ner']) or\
                (len(par['predictions']['ner'][default_trigger]) == 0):

                continue

            desc_spans = par['predictions']['ner'][default_trigger]

            toks = par['text'].split(' ')
            for i, j in desc_spans:
                toks.insert(j, '[P2]')
                toks.insert(i, '[P1]')
                par_txt = '\n'.join(toks)
                toks.pop(i)
                toks.pop(j)
                
                bio_pars.append(par_txt)

        return '\n\n'.join(bio_pars)
