from ultralytics import YOLO
import os
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from collections import defaultdict
from utils import identify_nouns_pos_and_synsets, find_pos

#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
#nltk.download('omw-1.4')



def predict_objects_pos_and_size(model, img,pict_dir):
    img_path = os.path.join(pict_dir, img)
    results = model(img_path)
    d_objects = defaultdict(int)

    #results[0].show()

    for result in results:
        for box in result.boxes:
            entity = result.names[int(box.cls)]  
            bbox = box.xyxyn[0].tolist()  # Get normislized bounding box coordinates [x1, y1, x2, y2]
            size = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
            d_objects[entity]+=size

    return d_objects





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
                l_categ = d_size.keys()
                categ = find_pos(syn,l_categ)
                #print("categ",categ)
                #store img, word, categ, size, position, 
                for syn in categ:
                    pos = noun_pos[word]
                    #for pos in l_pos:
                    l_word.append(word)
                    l_position.append(pos)
                    l_image.append(img)
                    l_caption.append(sentence)
                    l_label.append(syn)
                    l_size.append(d_size[syn])
                    l_detected_obj.append(d_size.keys())
    df = pd.DataFrame({'image': l_image, "detetcted_obj":l_detected_obj,'caption': l_caption, 'word': l_word,"position":l_position,"label":l_label,"size":l_size})
    df.to_csv('output.csv', index=False)




def prop_of_detected_w_are_mentionned(caption_path,pict_dir,obj_detection_model,nb_img,tokenizer=None,model=None,wsd=False):
    #read data cation file
    df = pd.read_csv(caption_path, delimiter="|", encoding="utf-8") 
    l_img_cap = img_capt_dict(df,nb_img)

    d_mentionned=defaultdict(list)

    for d in l_img_cap:
        img = d["image_name"]
        l_captions = d["captions"]
        #print("capt",l_captions)
        #get objects and size
        d_size = predict_objects_pos_and_size(obj_detection_model,img,pict_dir)
        #print("d size",d_size)
        for sentence in l_captions:
            #get nouns for each caption
            noun_synset,noun_pos = identify_nouns_pos_and_synsets(sentence,tokenizer,model,wsd)
            for word,syn in noun_synset.items():
                l_categ = d_size.keys()
                categ = find_pos(syn,l_categ)
                #print("syn and cat",syn,categ)
                is_ment = 0
                if categ != []:
                    is_ment = 1
                d_mentionned[word].append(is_ment)
 
    d_out = {k:sum(v)/len(v) for k,v in d_mentionned.items()}
    return d_out


def prop_of_mentionned_w_are_detected(caption_path,pict_dir,obj_detection_model,nb_img,wsd=False):
    #read data cation file
    df = pd.read_csv(caption_path, delimiter="|", encoding="utf-8") 
    l_img_cap = img_capt_dict(df,nb_img)

    d_mentionned=defaultdict(list)

    for d in l_img_cap:
        img = d["image_name"]
        l_captions = d["captions"]
        #print("capt",l_captions)
        #get objects and size
        d_size = predict_objects_pos_and_size(obj_detection_model,img,pict_dir)
        #print("d size",d_size)
        for sentence in l_captions:
            #get nouns for each caption
            noun_synset,noun_pos = identify_nouns_pos_and_synsets(sentence,wsd)
            for word,syn in noun_synset.items():
                l_categ = d_size.keys()
                categ = find_pos(syn,l_categ)
                #print("syn and cat",syn,categ)
                is_ment = 0
                if categ != []:
                    is_ment = 1
                d_mentionned[word].append(is_ment)
 
    d_out = {k:sum(v)/len(v) for k,v in d_mentionned.items()}
    return d_out


if __name__ == "__main__":

    pict_dir = "flickr30k_images/flickr30k_images"
    l_pict = os.listdir(pict_dir)
    img = l_pict[1]


    wsd = False
    img_model = True

    if wsd:
        model = AutoModelForSeq2SeqLM.from_pretrained("jpelhaw/t5-word-sense-disambiguation")
        tokenizer = AutoTokenizer.from_pretrained("jpelhaw/t5-word-sense-disambiguation")

        test_sentence = "A dog is playing in the park with a ball."
        noun_synsets, nouns_pos = identify_nouns_pos_and_synsets(test_sentence,tokenizer,model,wsd)
        #print(noun_synsets)


    if img_model:
        #model = YOLO("yolo11n.pt")
        obj_detection_model = YOLO("yolov8s-world.pt")
        caption_path = "flickr30k_images/results.csv"
        nb_img = 3

        prop_of_detected_w_are_mentionned(caption_path,pict_dir,obj_detection_model,nb_img,tokenizer,model,wsd)

    #get_syn_pos_size_df(caption_path,10)

    #df = pd.read_csv("flickr30k_images/results.csv", delimiter="|", encoding="utf-8") 
    #sentence = "a girl i klimbing a rock"
    #d_size = predict_objects_pos_and_size(model, img,pict_dir)
    #d_cap = img_capt_dict(df,10)
    #print(d_size)
    #noun_synset = identify_nouns_and_synsets(sentence,False)
    #print(noun_synset)
    #for word,syn in noun_synset.items():
    #    categ = find_pos(syn,d_size)
    #    print("categ",categ)

