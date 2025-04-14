

from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import defaultdict
import pandas as pd

def identify_nouns_pos_and_synsets(sentence,tokenizer=None,model=None,wsd=False):
    """Identify nouns in a sentence and associate them with WordNet synsets using T5 WSD model."""
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    
    noun_synsets = {}
    noun_pos = defaultdict(int)
    noun_order = defaultdict(int)
    noun_pos_rel = defaultdict(int)
    noun_order_rel = defaultdict(int)
    len_sent = len(tagged)
    order = 0
    tot_noun = sum([1 for (word, tag) in tagged if tag in ('NN', 'NNS')])
    for i,(word, tag) in enumerate(tagged):
        if tag in ('NN', 'NNS'):
            order+=1
            if wsd:
                descriptions = [syn.definition() for syn in wn.synsets(word, 'n')]
                if not descriptions:
                    continue
                input_text = f'question: which description describes the word "{word}" best in the following context? '
                input_text += f'descriptions: {descriptions} context: {sentence}'
                inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=True, padding=True)
                output = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=135)
                best_definition = tokenizer.decode(output[0], skip_special_tokens=True)
                for synset in wn.synsets(word, 'n'):
                    if synset.definition() == best_definition:
                        noun_synsets[word] = synset
                        break
            else:

                l_syn = wn.synsets(word, 'n')
                if len(l_syn)>0:
                    syn = l_syn[0]
                    if syn not in noun_pos.keys(): #consider only the first occurence of a synset
                        noun_synsets[word] = syn
                        noun_pos[syn]=(i+1)
                        noun_order[syn]=order
                        noun_pos_rel[syn]=(i+1)/len_sent
                        noun_order_rel[syn]=order/tot_noun
    
    return noun_synsets, noun_pos, noun_order, noun_pos_rel,noun_order_rel

def find_pos(syn,s_categ):
    hyp_path = [elem.name()[:-5] for elem in syn.hypernym_paths()[0]]
    categ = s_categ & set(hyp_path)
    return list(categ)



def group_and_avg(df):
    # Create 10 bins based on 'size'
    df['size_bin'] = pd.qcut(df['size'], q=10, duplicates='drop')  # or use pd.cut for equal-width bins

    # Group by label and size_bin, then average 'order'
    grouped = df.groupby(['label', 'size_bin'])['order'].mean().reset_index()

    # Optional: rename for clarity
    grouped = grouped.rename(columns={'order': 'avg_order'})

    return grouped
