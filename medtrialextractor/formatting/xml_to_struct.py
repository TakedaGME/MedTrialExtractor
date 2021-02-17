'''
File to host parsing logic.
'''

import os
import uuid
import re
import json
import codecs
from bs4 import BeautifulSoup
from spacy.lang.en import English
from multiprocessing import Pool

# ----- Utils ----- #
nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)


def tokenize(text):
    text = ' '.join([t.text for t in tokenizer(text)])
    text = re.sub('\n+', '\n', text)
    text = re.sub('[ ]{2,}', ' ', text)
    text = '\n'.join([s.strip() for s in text.split('\n') if s.strip()])
    return text

# ----- XML Parser -------- #


def parse_xml(file_path):
    with open(file_path, 'r') as f:
        soup = BeautifulSoup(f.read(), features="html.parser")

        # No producer ID for now, just use a random UUID
        doc_id = str(uuid.uuid4())
        file_name = os.path.basename(file_path).replace('.xml', '')

        doc_struct = {
            'document_id': doc_id,
            'file_name': file_name,
            'paragraphs': [],
        }

        ## File Description Data ##
        doc_desc = soup.find('filedesc')

        title = ''
        for t in doc_desc.find_all('title'):
            title += t.text + ' '
        title = tokenize(title.strip())
        doc_struct['title'] = title

        raw_desc = ''
        for c in doc_desc.find_all():
            if c.text:
                raw_desc += ' ' + c.text.strip()
        raw_desc = tokenize(raw_desc)

        # Join all < 3 word paragraphs
        desc_paragraphs = [None]
        for p in raw_desc.split('\n'):
            if len(p.split(' ')) <= 3:
                if desc_paragraphs[-1]:
                    desc_paragraphs[-1] += ' ' + p
                else:
                    desc_paragraphs[-1] = p
            else:
                if desc_paragraphs[-1]:
                    desc_paragraphs.append(p)
                else:
                    desc_paragraphs[-1] = p
                    desc_paragraphs.append(None)
        if not desc_paragraphs[-1]:
            desc_paragraphs.pop()

        # Add to data
        for p in desc_paragraphs:
            doc_struct['paragraphs'].append({
                'text': p,
                'head': 'DOCUMENT DESCRIPTION',
                'position': 'DOCUMENT DESCRIPTION',
                'annotated': False,
            })

        ## Abstract ##
        abstract = soup.find('abstract')
        if abstract:
            abstract_raw = ''
            for p in abstract.find_all():
                abstract_raw += ' ' + p.text
            abstract_raw = tokenize(abstract_raw)
            doc_struct['paragraphs'].append({
                'text': abstract_raw,
                'head': 'ABSTRACT',
                'position': 'ABSTRACT',
                'annotated': False,
            })

        ## Main Body ##
        position = 0
        body = soup.find('text').find('body')
        for div in body.find_all('div', recursive=False):
            head = ''
            for h in div.find_all('head'):
                head += ' ' + h.text

            for p in div.find_all('p'):
                ptext = p.text
                doc_struct['paragraphs'].append({
                    'text': tokenize(ptext),
                    'head': head.strip(),
                    'position': f'Paragraph {position}',
                    'annotated': False,
                })
                position += 1

        ## Get Authors ##

        author_list = []
        for author_div in doc_desc.find_all('author'):
            auth_name = []
            for fname in author_div.find_all('forename'):
                auth_name.append(fname.text)
            for sname in author_div.find_all('surname'):
                auth_name.append(sname.text)
            auth_name = ' '.join(auth_name)
            if len(auth_name) > 0:
                author_list.append(auth_name)
        doc_struct['authors'] = author_list

        return doc_struct


def batch_process(input_dir, output_file, **kwargs):
    input_files = [os.path.join(input_dir, fname) for fname in os.listdir(
        input_dir) if fname.lower().endswith('.xml')]
    threads = kwargs.get('num_workers', 1)

    if threads > 1:
        with Pool(threads) as pool:
            structs = pool.map(parse_xml, input_files)
    else:
        # Loop rather than pool to facilitate debugging
        structs = []
        for ifile in input_files:
            structs.append(parse_xml(ifile))

    doc_structs = {el['document_id']: el for el in structs}
    structs = {
        'documents': doc_structs,
    }

    # Make directories to output file if they don't already exist.
    base_dir = os.path.dirname(output_file)
    os.makedirs(base_dir, exist_ok=True)

    # Save output struct
    with codecs.open(output_file, 'wb', encoding='utf-8', errors='replace') as outfile:
        json.dump(structs, outfile, indent=4)

def xml_batch_process_cli(args):

    kwargs = vars(args)
    input_dir = os.path.realpath(args.xml_dir)
    output_file = os.path.realpath(args.output_path)
    kwargs['input_dir'] = input_dir
    kwargs['output_file'] = output_file

    batch_process(**kwargs)





