'''
Read Annotations file and save in struct
'''

import pandas as pd
import json
import codecs
import multiprocessing
from contextlib import contextmanager


# Utils
def match_ids_by_filename(ann, struct):
    fname_to_id = {}
    for doc_id in pd.unique(ann['doc_id']):
        doc_name = ann.loc[ann['doc_id'] == doc_id].iloc[0]['doc_name']
        fname_to_id[doc_name] = doc_id
    
    new_struct = struct.copy()
    new_struct['documents'] = {}

    for old_id in struct['documents']:

        doc_struct = struct['documents'][old_id].copy()
        fname = doc_struct['file_name']
        new_id = fname_to_id.get(fname, old_id)
        doc_struct['document_id'] = new_id

        new_struct['documents'][new_id] = doc_struct
    
    
    return new_struct

def get_spans(spans_str):
    idxs = [int(e) for e in spans_str.split(',') if len(e) > 0]
    assert(len(idxs)%2 == 0)
    return list(zip(idxs[::2], idxs[1::2]))

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

# Main 
def read_annotations(args):

    # Read input files
    annotation_file_path = args.annotation_file_path
    struct_base_path = args.struct_base_path

    annotations = pd.read_csv(annotation_file_path, keep_default_na=False)

    struct_base = None
    with codecs.open(struct_base_path, 'r', encoding='utf-8', errors='replace') as struct_base_file:
        struct_base = json.load(struct_base_file)

    # Spacing for output readability
    print('\n' * 2)

    # Match ids by file name if flag present
    if args.match_ids_by_filename:
        print('Matching document ids by file name ...')
        struct_base = match_ids_by_filename(annotations, struct_base)
        print('done.\n')

    struct_doc_ids = set(struct_base['documents'].keys())
    annotated_doc_ids = set(pd.unique(annotations['doc_id']))

    annotated_docs = annotated_doc_ids & struct_doc_ids
    not_annotated_docs = struct_doc_ids - annotated_doc_ids
    unknown_annotated_docs = annotated_doc_ids - struct_doc_ids

    if args.verbose:
        if len(annotated_docs) > 0:
            print('Annotated Documents:')
            for j, id in enumerate(annotated_docs):
                print(f'{j + 1}. {id}')
            print()
            print('-' * 100)
        else:
            print(f'No annotations for documents in {struct_base_path} were found.')
            print()
            print('-' * 100)

        if len(not_annotated_docs) > 0:
            print('No annotations found for the following documents:')
            for j, id in enumerate(not_annotated_docs):
                print(f'{j + 1}. {id}')
            print()
            print('-' * 100)

        if len(unknown_annotated_docs) > 0:
            print('These files were found in the annotaiton file but are not present in the base struct file:')
            print('No annotations found for the following documents:')
            for j, id in enumerate(unknown_annotated_docs):
                print(f'{j + 1}. {id}')
            print()
            print('-' * 100)

    # for doc_id in annotated_docs:
    annotation_args = [(struct_base['documents'][id], annotations) for id in annotated_docs]

    with poolcontext(processes=args.num_workers) as pool:
        doc_structs = pool.starmap(read_document_annotation, annotation_args)
        for ds in doc_structs:
            id = ds['document_id']
            struct_base['documents'][id] = ds

    if 'annotated_docs' not in struct_base:
        struct_base['annotated_docs'] = []
    struct_base['annotated_docs'].extend(list(annotated_docs))

    # Save annotated struct
    if args.struct_annotated_path is not None:
        struct_annotated_path = args.struct_annotated_path
    else:
        struct_annotated_path = args.struct_base_path

    with codecs.open(struct_annotated_path, 'w', encoding='utf-8', errors='replace') as output_struct:
        json.dump(struct_base, output_struct, indent=4)

def read_document_annotation(doc_struct, annotations):

    doc_id = doc_struct['document_id']
    doc_df = annotations.loc[annotations['doc_id'] == doc_id]

    print(f'Processing annotations for {doc_id} ...')

    if len(doc_df) == 0:
        raise Exception(f'Unable to find annotations for document: {doc_id}')

    non_arm_cols = ['title', 'authors', 'study_type']
    arm_non_numbered = ['arm_efficacy_metric', 'arm_efficacy_results']
    arm_numbered  = ['arm_dosage', 'arm_description']

    # Iterate over doc_struct paragraphs
    for par in doc_struct['paragraphs']:
        ann_par = doc_df.loc[doc_df['description'] == par['text']]

        for j, row in ann_par.iterrows():
            par['annotated'] = True
            par['annotations_spans'] = {}
            par['annotations_span_arms'] = {}

            # Get arm number
            arm_count = int(row['arm_number'])

            # Get annotations for non-arm columns
            for k in non_arm_cols:
                for kp in row.index:
                    if k in kp:
                        ann_spans = par['annotations_spans']
                        ann_span_arms = par['annotations_span_arms']

                        tags_col = f'{k}-tag'
                        span_str = row[tags_col]
                        spans = get_spans(span_str)

                        ann_spans[k] = spans
                        ann_span_arms = [('*', '*')] * len(spans)

                        # Don't duplicate cols
                        break


            # Get annotations for arm non-numbered columns
            for k in arm_non_numbered:
                for kp in row.index:
                    if k in kp:
                        ann_spans = par['annotations_spans']
                        ann_span_arms = par['annotations_span_arms']

                        tags_col = f'{k}-tag'
                        span_str = row[tags_col]
                        spans = get_spans(span_str)

                        ann_spans[k] = spans
                        ann_span_arms[k] = [(arm_count, '*')] * len(spans)

                        # Don't duplicate cols
                        break

            # Get annotations for arm columns
            for k in arm_numbered:
                for kp in row.index:
                    if k in kp:
                        ann_spans = par['annotations_spans']
                        ann_span_arms = par['annotations_span_arms']

                        desc_idxs = []
                        for c in row.index:
                            if c.startswith(k) and c.endswith('-tag'):
                                c_idx = int(c[len(k) + 1:-4])
                                desc_idxs.append(c_idx)

                        k_spans = []
                        k_arms = []
                        
                        for d_idx in desc_idxs:
                            tag_col = f'{k}-{d_idx}-tag'
                            span_str = row[tag_col]
                            d_spans = get_spans(span_str)
                            d_arms = [(arm_count, d_idx)] * len(d_spans)
                            k_spans.extend(d_spans)
                            k_arms.extend(d_arms)

                        ann_spans[k] = k_spans
                        ann_span_arms[k] = k_arms

                        # Don't duplicate cols
                        break

    return doc_struct