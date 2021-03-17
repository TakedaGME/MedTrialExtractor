import json
import re
from struct_to_bio import struct_to_bio_dict, struct_to_bio_dict_rd
from seqeval.metrics.sequence_labeling import get_entities

def make_empty_ner_bio(struct_path, output_path):
    bio_dir = struct_to_bio_dict(struct_path, 1, use_tags=False)

    paragraphs = []

    for doc_id in bio_dir:
        pars = bio_dir[doc_id].strip().split('\n\n')

        for j, par in enumerate(pars):
            par_txt = f'#\tpassage={doc_id}\tpar_idx={j}\n'
            par_txt += par
            par_txt = par_txt.strip()
            paragraphs.append(par_txt)

    ner_input = '\n\n'.join(paragraphs)

    with open(output_path, 'w') as ner_input_file:
        ner_input_file.write(ner_input)


def load_ner_predictions(struct_path, ner_output_file_path, output_struct_path):

    # Load base struct

    struct = None
    with open(struct_path, 'r') as struct_file:
        struct = json.load(struct_file)

    # Load annotation output
    ner_txt = None
    with open(ner_output_file_path, 'r') as ner_output_file:
        ner_txt = ner_output_file.read()

    extracted_paragraphs = [par for par in ner_txt.split('\n\n') if len(par.strip()) > 0]

    # Parse and add predictions to struct
    for par in extracted_paragraphs:
        lines = [l for l in par.split('\n') if len(l.strip()) > 0]

        # parse paragraph info
        par_desc = lines[0].strip()
        assert(par_desc.startswith('#'))

        par_desc = re.sub('\s+|\t', ' ', par_desc).split(' ')[1:]
        par_info = dict([e.split('=') for e in par_desc])

        document_id = par_info['passage']
        paragraph_idx = int(par_info['par_idx'])

        tokens, labels = zip(*[re.sub('\s+|\t', ' ', e.strip()).split() for e in lines[1:]])
        tokens = list(tokens)
        labels = list(labels)

        ents = get_entities(labels)
        spans = {}
        for k, start, stop in ents:
            if k not in spans:
                spans[k] = []
            spans[k].append((start, stop + 1))
        
        par_dict = struct['documents'][document_id]['paragraphs'][paragraph_idx]
        if 'predictions' not in par_dict:
            par_dict['predictions'] = dict()
        
        par_dict['predictions']['ner'] = spans
    
    with open(output_struct_path, 'w') as output_struct_file:
        json.dump(struct, output_struct_file, indent=4)

def make_empty_rd_input(struct_path, output_path):
    bio_dir = struct_to_bio_dict_rd(struct_path, is_pred=True)

    paragraphs = []

    for doc_id in bio_dir:
        pars = bio_dir[doc_id].strip().split('\n\n')

        for j, par in enumerate(pars):
            par_txt = f'#\tpassage={doc_id}\tpar_idx={j}\n'
            par_txt += par
            par_txt = par_txt.strip()
            paragraphs.append(par_txt)

    ner_input = '\n\n'.join(paragraphs)

    with open(output_path, 'w') as ner_input_file:
        ner_input_file.write(ner_input)

if __name__ == '__main__':

    struct_base_path = '/data/rsg/nlp/juanmoo1/projects/05_dev/workdir/collections_dir/Hello/structs/struct_base.json'
    ner_output_path = '/data/rsg/nlp/juanmoo1/projects/05_dev/workdir/tempdir/root/ner_output.txt'
    output_struct_path = '/data/rsg/nlp/juanmoo1/projects/05_dev/workdir/collections_dir/Hello/structs/struct_pred.json'

    bio_output_path = '/data/rsg/nlp/juanmoo1/projects/05_dev/workdir/tempdir/root/rd_input.txt'

    make_empty_rd_input(output_struct_path, bio_output_path)


    # load_ner_predictions(struct_base_path, ner_output_path, output_struct_path) 