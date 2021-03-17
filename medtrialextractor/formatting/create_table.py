'''
Create summary table from NER and RD predictions
'''

import json
from os import kill
import pandas as pd


def create_table(args):

    # Load struct
    struct = None
    with open(args.input_struct, 'r') as struct_file:
        struct = json.load(struct_file)

    document_ids = list(struct['documents'].keys())

    # Get NER entities
    ner_entities = {id: [] for id in document_ids}
    rd_entities = {id: [] for id in document_ids}

    for doc_id in document_ids:
        doc_struct = struct['documents'][doc_id]

        for par in doc_struct['paragraphs']:
            toks = par['text'].split(' ')

            if 'predictions' in par:
                if 'ner' in par['predictions']:
                    ner_preds = par['predictions']['ner']
                    num_ents = sum([len(e) for e in ner_preds.values()])
                    if num_ents > 0:
                        new_dict = {}
                        for k in ner_preds:
                            new_dict[k] = []
                            for i, j in ner_preds[k]:
                                new_dict[k].append(' '.join(toks[i:j]))
                            
                            new_dict[k] = ' | '.join(new_dict[k])

                        ner_entities[doc_id].append(new_dict)

                if 'rd' in par['predictions']:
                    rd_preds = par['predictions']['rd']

                    for rd_pred in rd_preds:
                        num_ents = sum([len(e) for e in rd_pred.values()])
                        if num_ents > 0:
                            new_dict = {}
                            for k in rd_pred:
                                new_dict[k] = []
                                for i, j in rd_pred[k]:
                                    new_dict[k].append(' '.join(toks[i:j]))
                                new_dict[k] = ' | '.join(new_dict[k])
                            rd_entities[doc_id].append(new_dict)

    # Create Table
    dfs = []
    target_cols = ['document_id', 'document_name', 'title', 'authors', 'study_type',
                   'arm_description', 'arm_dosage', 'arm_efficacy_metric', 'arm_efficacy_results']
    for doc_id in rd_entities:

        if doc_id not in rd_entities:
            continue

        doc_ents = rd_entities[doc_id]
        doc_df = pd.DataFrame(doc_ents)
        for col in target_cols:
            if col not in doc_df:
                doc_df[col] = ''
        doc_df = doc_df[target_cols]

        # Place non-arm information in the first line
        authors = struct['documents'][doc_id]['authors'][:5]
        study_type = ['placeholder']
        doc_name = struct['documents'][doc_id]['file_name']
        title = struct['documents'][doc_id]['title']

        if len(doc_df) > 0:
            doc_df.iloc[0]['authors'] = ' || '.join(authors)
            doc_df.iloc[0]['study_type'] = ' || '.join(study_type)
            doc_df.iloc[0]['document_id'] = doc_id
            doc_df.iloc[0]['document_name'] = doc_name
            doc_df.iloc[0]['title'] = title
        else:
            doc_df.append({
                'authors': ' || '.join(authors),
                'study_type': ' || '.join(study_type),
                'document_id': doc_id,
                'document_name': doc_name,
                'title': title
            }, ignore_index=True)
        dfs.append(doc_df)

    df = pd.concat(dfs)
    # Colum Renames
    rename_map = {'document_id': 'Document ID', 'document_name': 'File Name', 'title': 'Title', 'authors': 'Authors', 'study_type': 'Study Type',
                  'arm_description': 'Description', 'arm_dosage': 'Dosage', 'arm_efficacy_metric': 'Metric',
                  'arm_efficacy_results': 'Efficacy Results'}
    df = df.rename(columns=rename_map, errors='raise')

    # Save Table
    df.to_excel(args.output_path, index=False)
    return df
