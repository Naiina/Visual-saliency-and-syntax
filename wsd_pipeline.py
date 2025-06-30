import nltk
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import string
from nltk.corpus import wordnet as wn
import nltk
from nltk.tag import pos_tag
from collections import defaultdict
import pandas as pd
from nltk.tokenize import MWETokenizer
from pycocotools.coco import COCO
import os
from utils import convert_loc_file_into_dict, get_coco_capt
from tqdm import tqdm
import json

from nltk.tokenize import MWETokenizer
import torch._dynamo
#torch._dynamo.config.cache_size_limit = 64 


mwe_phrases = [
    ('sports', 'ball'),
    ('baseball', 'bat'),
    ('baseball', 'glove'),
    ('tennis', 'racket'),
    ('wine', 'glass'),
    ('hot', 'dog'),
    ('potted', 'plant'),
    ('dining', 'table'),
    ('cell', 'phone'),
    ('teddy', 'bear'),
    ('hair', 'drier')
]

mwe_tokenizer = MWETokenizer(mwe_phrases, separator='_')


def best_matching_def(llm_out: str, synset_to_def) -> str:
    def tokenize(sentence):
        return set(word.lower() for word in sentence.split())

    set_llm_out = tokenize(llm_out)
    best_match = None
    max_overlap = -1

    for syn, definition in synset_to_def.items():
        overlap = len(set_llm_out & tokenize(definition))
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = syn
    return best_match


def wsd_llm(word, sentence, tokenizer, model, device):
    synsets = wn.synsets(word, pos=wn.NOUN)
    d_synset_to_def = {syn: syn.definition() for syn in synsets}
    messages = build_wsd_template(word, sentence, d_synset_to_def)

    # Flatten messages into plain text prompt for LLM
    flat_prompt = ""
    for turn in messages[0]:
        role = turn["role"]
        for part in turn["content"]:
            flat_prompt += f"{role}: {part['text']}\n"
    flat_prompt += "assistant:"

    # Tokenize and generate
    inputs = tokenizer(flat_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract only model response (after "assistant:" tag)
    if "assistant:" in output_text:
        out_sent = output_text.split("assistant:")[-1].strip()
    else:
        out_sent = output_text.strip()

    syn = best_matching_def(out_sent, d_synset_to_def)
    return syn


def build_wsd_template(word, sentence, synset_to_def):
    def_str = ", ".join(f'"{elem}"' for elem in synset_to_def.values())

    prompt = (
        f'question: which description describes the word "{word}" '
        f'best in the following context?\n'
        f'descriptions: [ {def_str} ]\n'
        f'context: {sentence}\n'
    )

    messages = [[
        {
            "role": "system",
            "content": [{"type": "text", "text": 'Always only respond with the definition of the best matching sense based on context.'}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        },
    ]]
    return messages


def get_noun_caption(caption,mwe_tokenizer,wsd_tok=None,wsd_model=None,wsd=False,device="cpu"):
    l_nouns = []
    l_syn = []
    tokens = mwe_tokenizer.tokenize(caption.split())
    punct_without_underscore = string.punctuation.replace("_", "")
    tokens = [token.translate(str.maketrans('', '', punct_without_underscore)) for token in tokens]
    tagged = pos_tag(tokens)
    for i,(word, tag) in enumerate(tagged):
        if tag in ('NN', 'NNS'):
          if wsd:
                syn = wsd_llm(word, caption,wsd_tok, wsd_model, device)
          else:
                print("syn",syn)
                if len(syn)>0:
                    syn = wn.synsets(word, 'n')[0]
                else:
                    syn = None
          if syn is not None:
            l_nouns.append(word)
            l_syn.append(syn.name())
    return l_nouns,l_syn



def main(loc_n_jsonl,dataset_cap,k,mwe_tokenizer,wsd_tok=None,wsd_model=None,wsd=False,device="cpu"):
    d_loc_n_cations = convert_loc_file_into_dict(loc_n_jsonl)
    d_all = {}
    
    for it,id_img in enumerate(tqdm(d_loc_n_cations.keys())):
        if k>0 and it >k:
            break
        img = coco.loadImgs([int(id_img)])[0]
        l_info = []

        if dataset_cap == "coco":
            l_captions = get_coco_capt(img,coco_caps)
        if dataset_cap == "localized_narratives":
            l_captions = d_loc_n_cations[id_img]
        
        for caption in l_captions:
            l_nouns,l_syn = get_noun_caption(caption,mwe_tokenizer,wsd_tok,wsd_model,wsd,device)
            l_info.append([caption,l_nouns,l_syn])
        d_all[id_img] = l_info
        if it%50 == 0:
            torch._dynamo.reset()
            with open("intermediate_save_synsets.json", "w") as f:
                json.dump(d_all, f, indent=4)
    return d_all




if __name__ == "__main__":
    #nltk.download('omw-1.4')
    #nltk.download('wordnet')

    
    max_it = -1
    localized_n_file = "localized_narratives/coco_val_captions.jsonl"
    wsd = True

    dataset_cap = "localized_narratives"
    save_file = dataset_cap+'_caption_syn_output.json'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model_name = "google/gemma-3-4b-it"
    wsd_tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    wsd_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True
    )
    dataDir='COCO'
    dataType='val2017'
    imgDir = os.path.join(dataDir, dataType)

    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
    annFile_cap = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
    coco=COCO(annFile)
    coco_caps=COCO(annFile_cap)

    d = main(localized_n_file,dataset_cap,max_it,mwe_tokenizer,wsd_tok,wsd_model,wsd,device)
    with open(dataset_cap+"_synsets.json", "w") as f:
        json.dump(d, f, indent=4)
