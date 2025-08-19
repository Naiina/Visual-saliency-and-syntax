from ultralytics import YOLO
import os
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from collections import defaultdict

#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
#nltk.download('omw-1.4')



def predict_objects_pos_and_size(model, img,pict_dir):
    img_path = os.path.join(pict_dir, img)
    results = model(img_path)
    d_objects = defaultdict(int)

    results[0].show()

    for result in results:
        for box in result.boxes:
            entity = result.names[int(box.cls)]  
            bbox = box.xyxyn[0].tolist()  # Get normislized bounding box coordinates [x1, y1, x2, y2]
            size = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
            d_objects[entity]+=size

    return d_objects


def find_pos(syn,d_size):
    hyp_path = [elem.name()[:-5] for elem in syn.hypernym_paths()[0]]
    categ = set(d_size.keys()) & set(hyp_path)
    return list(categ)


def img_capt_dict(df,nb_img):

    image_captions = {}
    for _,row in df.iterrows():

        image_name = row["image_name"]
        caption = row[" comment"]

        if image_name not in image_captions:
            if len(image_captions) >= nb_img:
                break  # Stop if we have reached the limit
            image_captions[image_name] = []
        
        image_captions[image_name].append(caption)
    
    return [{"image_name": k, "captions": v} for k, v in image_captions.items()]



def identify_nouns_pos_and_synsets(sentence,wsd=False):
    """Identify nouns in a sentence and associate them with WordNet synsets using T5 WSD model."""
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    
    noun_synsets = {}
    noun_pos = {}
    len_sent = len(tagged)
    for i,(word, tag) in enumerate(tagged):
        if tag in ('NN', 'NNS'):
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
                    if word in ["men","man","Men","Man"]:
                        syn = l_syn[1]
                    else:
                        syn = l_syn[0]
                    if word not in noun_pos.keys(): #keep just first one
                        noun_synsets[word] = syn
                        noun_pos[word] = (i+1)/len_sent
    
    return noun_synsets, noun_pos



def get_cap_frequ(caption_path,nb_img,wsd=False):

    df = pd.read_csv(caption_path, delimiter="|", encoding="utf-8") 
    l_img_cap = img_capt_dict(df,nb_img)
    d_freq_syn = defaultdict(int)
    for d in l_img_cap:
        l_captions = d["captions"]
        for sentence in l_captions:
            noun_synset,_ = identify_nouns_pos_and_synsets(sentence,wsd)
            for syn in noun_synset.values():
                d_freq_syn[syn]+=1
    return d_freq_syn





def get_syn_pos_size_df(caption_path,pict_dir,obj_detection_model,nb_img,wsd=False):
    #read data cation file
    df = pd.read_csv(caption_path, delimiter="|", encoding="utf-8") 
    l_img_cap = img_capt_dict(df,nb_img)

    l_word = []
    l_position = []
    l_image = []
    l_caption = []
    l_label = []
    l_size = []
    l_detected_obj = []

    for d in l_img_cap:
        img = d["image_name"]
        l_captions = d["captions"]
        #get objects and size
        d_size = predict_objects_pos_and_size(obj_detection_model,img,pict_dir)
        for sentence in l_captions:
            #get nouns for each caption
            noun_synset,noun_pos = identify_nouns_pos_and_synsets(sentence,wsd)
            for word,syn in noun_synset.items():
                categ = find_pos(syn,d_size)
                #print("categ",categ)
                #store img, word, categ, size, position, 
                for syn in categ:
                    pos = noun_pos[word]
                    l_word.append(word)
                    l_position.append(pos)
                    l_image.append(img)
                    l_caption.append(sentence)
                    l_label.append(syn)
                    l_size.append(d_size[syn])
                    l_detected_obj.append(d_size.keys())
    df = pd.DataFrame({'image': l_image, "detetcted_obj":l_detected_obj,'caption': l_caption, 'word': l_word,"position":l_position,"label":l_label,"size":l_size})
    df.to_csv('output.csv', index=False)




def prop_of_detected_w_are_mentionned(caption_path,pict_dir,obj_detection_model,nb_img,wsd=False):
    #read data cation file
    df = pd.read_csv(caption_path, delimiter="|", encoding="utf-8") 
    l_img_cap = img_capt_dict(df,nb_img)

    d_mentionned=defaultdict(list)
    d_ment_size=defaultdict(list)

    for d in l_img_cap:
        img = d["image_name"]
        l_captions = d["captions"]
        #get objects and size
        d_size = predict_objects_pos_and_size(obj_detection_model,img,pict_dir)
        for sentence in l_captions:
            #get nouns for each caption
            noun_synset,noun_pos = identify_nouns_pos_and_synsets(sentence,wsd)
            for word,syn in noun_synset.items():
                categ = find_pos(syn,d_size)
                is_ment = 0
                if categ != []:
                    is_ment = 1
                    d_ment_size[syn].append(d_size[categ[0]])
                d_mentionned[syn].append(is_ment)
    print(d_mentionned)          
    d_out = {k:sum(v)/len(v) for k,v in d_mentionned.items() }
    
    d_out_size = {k:sum(v)/len(v) for k,v  in d_ment_size.items() }
    return d_out,d_out_size




if __name__ == "__main__":

    pict_dir = "../picture_saliency/flickr30k_images/flickr30k_images"
    caption_path = "../picture_saliency/flickr30k_images/results.csv"

    wsd = False
    img_model = False

    if wsd:
        model = AutoModelForSeq2SeqLM.from_pretrained("jpelhaw/t5-word-sense-disambiguation")
        tokenizer = AutoTokenizer.from_pretrained("jpelhaw/t5-word-sense-disambiguation")

        test_sentence = "A dog is playing in the park with a ball."
        noun_synsets = identify_nouns_pos_and_synsets(test_sentence)

    if img_model:
        obj_detection_model = YOLO("yolov8s-world.pt")
        nb_img = 3
        prop_of_detected_w_are_mentionned(caption_path,pict_dir,obj_detection_model,nb_img,wsd)

