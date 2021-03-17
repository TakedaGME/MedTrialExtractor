import json
from . import struct_to_bio_dict

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


def load_ner_predictions(struct_path, ner_output_file, output_struct_path):

    struct = None
    with open(struct_path, 'r') as struct_file:
        struct = json.load(struct_file)

    
if __name__ == '__main__':

    struct_base_path = 