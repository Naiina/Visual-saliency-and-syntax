import os
import random
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import defaultdict
import numpy as np
import seaborn as sns
import pandas as pd


from nltk.corpus import wordnet as wn
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer

#import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

#random.seed(44)




def number_info(coco,img_id):
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    category_counts = defaultdict(list)

    for ann in anns:
        cat_id = ann['category_id']
        iscrowd = ann['iscrowd']
        category_counts[cat_id].append(iscrowd)

    print(f"\nImage ID: {img_id}")
    print("Categories in image and their types:")
    for cat_id, iscrowds in category_counts.items():
        cat_name = coco.loadCats(cat_id)[0]['name']
        if all(c == 1 for c in iscrowds):
            print(f" - {cat_name}: crowd")
        elif len(iscrowds) == 1 and iscrowds[0] == 0:
            print(f" - {cat_name}: single entity")
        else:
            print(f" - {cat_name}: several entities")


def count_sing_plur_crowd(coco):
    
    l_all_categ = coco.loadCats(coco.getCatIds())
    d_id_categ_to_categ = {elem["id"]:elem["name"] for elem in l_all_categ}
    d_id_categ_to_super_categ = {elem["id"]:elem["supercategory"] for elem in l_all_categ}
    l_img_id = coco.getImgIds()
    d_count_super_categ = defaultdict(lambda: defaultdict(int))
    for img_id in l_img_id:
        #display_picture(coco,img_id)
        d_crowd = defaultdict(int)
        d_sing = defaultdict(int)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for elem in anns:
            categ_id = elem["category_id"]
            categ = d_id_categ_to_categ[categ_id]
            super_categ = d_id_categ_to_super_categ[categ_id]
            #print(super_categ)
            crowd = elem["iscrowd"]
            if crowd:
                d_crowd[super_categ]+=1
            else:
                d_sing[super_categ]+=1
        #print(d_crowd)
        #print(d_sing)
        for k,v in d_sing.items():
            if v == 1:
                d_count_super_categ[k]["sing"]+=1
            else:
                d_count_super_categ[k]["plur"]+=1
        for k,v in d_crowd.items():
            d_count_super_categ[k]["crowd"]+=1
    #for k,v in d_count_super_categ.items():
    #    print(k,dict(v))
    
        
    return d_count_super_categ

def plot_histo_number(data):
    records = []
    for category, counts in data.items():
        for kind, value in counts.items():
            records.append({'category': category, 'type': kind, 'count': value})

    df = pd.DataFrame(records)

    # Plot
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='category', y='count', hue='type')

    plt.title("Sing/Plur/Crowd Distribution per Category")
    plt.ylabel("Count")
    plt.xlabel("Category")
    plt.legend(title="Type")
    plt.tight_layout()
    plt.show()


def find_picts_with_sing_and_crowd_person(coco,k=-1):
    iter = 0
    person_cat_id = coco.getCatIds(catNms=['person'])
    img_ids = coco.getImgIds(catIds=person_cat_id)
    
    matching_img_ids = []

    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=person_cat_id)
        anns = coco.loadAnns(ann_ids)

        has_single = any(ann['iscrowd'] == 0 for ann in anns)
        has_crowd = any(ann['iscrowd'] == 1 for ann in anns)

        if has_single and has_crowd:
            iter+=1
            matching_img_ids.append(img_id)
        if iter >k and k>0:
            return matching_img_ids
        
    return matching_img_ids


def get_caption(coco_caps, img_id):

    ann_ids = coco_caps.getAnnIds(imgIds=img_id)
    anns = coco_caps.loadAnns(ann_ids)
    l_cap = [ann['caption'] for ann in anns]
    return l_cap


def display_picture(coco,img_id):

    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(img_dir, img_info['file_name'])
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Image ID: {img_id}")
    plt.show()


def refers_to_person(word):
    synsets = wn.synsets(word, pos=wn.NOUN)
    if not synsets:
        return False
    main_synset = synsets[0] 
    #print(f"Synset: {main_synset.name()}: {main_synset.definition()}")

    return any('person.n.01' in str(hyper) for hyper in main_synset.hypernym_paths()[0])


def refers_to_a_group(word):
    synsets = wn.synsets(word, pos=wn.NOUN)
    if not synsets:
        return False
    main_synset = synsets[0] 
    #print(f"Synset: {main_synset.name()}: {main_synset.definition()}")

    return any('group.n.01' in str(hyper) for hyper in main_synset.hypernym_paths()[0])


def refers_to_supercategory(word, super_categ):

    synsets = wn.synsets(word, pos=wn.NOUN)
    if not synsets:
        return False
    main_synset = synsets[0]  # most common noun meaning
    print(main_synset, main_synset.hypernym_paths()[0])
    return any(super_categ+'.n.01' in str(hyper) for hyper in main_synset.hypernym_paths()[0])


def number_segmentation_of_human_nouns(sentence):

    singular_nouns = []
    plural_nouns = []

    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)

    for word, tag in tagged:
        if tag in ['NN', 'NNS']:  # NN: singular, NNS: plural
            lemma = lemmatizer.lemmatize(word, pos='n')
            if refers_to_a_group(word):
                plural_nouns.append(lemma)
            else:
                is_person = refers_to_supercategory(lemma)
                if is_person:
                    if tag == 'NNS':
                        plural_nouns.append(lemma)
                    else:
                        singular_nouns.append(lemma)
    return {"sing":singular_nouns, "plur":plural_nouns}


def get_k_random_image_ids_by_supercategory(coco, supercategory_name, k):
    
    cat_ids = coco.getCatIds()
    super_cat_ids = [cat['id'] for cat in coco.loadCats(cat_ids) if cat['supercategory'] == supercategory_name]

    if not super_cat_ids:
        raise ValueError(f"No categories found for supercategory: '{supercategory_name}'")

    img_ids = coco.getImgIds(catIds=super_cat_ids)

    if k > len(img_ids):
        raise ValueError(f"Requested {k} images, but only {len(img_ids)} found for supercategory '{supercategory_name}'.")

    return random.sample(img_ids, k)


def get_coco_supercategories(coco):
    cats = coco.loadCats(coco.getCatIds())
    supercategories = sorted(set(cat['supercategory'] for cat in cats))
    return supercategories

data_dir = 'COCO'
ann_file = os.path.join(data_dir, 'annotations/instances_val2017.json')
img_dir = os.path.join(data_dir, 'val2017')
coco = COCO(ann_file)


caption_ann_file = os.path.join(data_dir, 'annotations/captions_val2017.json')
coco_caps = COCO(caption_ann_file)


img_id = 188465

#img_id = find_pict_with_sing_and_crowd(coco)
#display_picture(coco,img_id)
#print_caption(coco_caps, img_id)





def starts_number(l_img_id):
    l_sing = []
    l_sev_sing = []
    l_plur = []
    for img_id in l_img_id:
        display_picture(coco,img_id)
        nb_cap_sing = 0
        nb_cap_sev_sing = 0
        nb_cap_plur = 0
        l_cap = get_caption(coco_caps, img_id)
        for cap in l_cap:
            print(cap)
            d_number = number_segmentation_of_human_nouns(cap,l_pseudo_plur)
            if len(d_number["sing"])>0:
                nb_cap_sing+=1
                if len(d_number["sing"])>1:
                    nb_cap_sev_sing+=1
            if len(d_number["plur"])>0:
                nb_cap_plur+=1
        l_sing.append(nb_cap_sing)
        l_sev_sing.append(nb_cap_sev_sing)
        l_plur.append(nb_cap_plur)
    return l_sing,l_sev_sing,l_plur




lemmatizer = WordNetLemmatizer()
get_coco_supercategories(coco)
l_pseudo_plur = ["crowd","team","poeple","family","group","committee","audience",
          	"class","staff","crew","jury","panel","band","choir","army","public","gang","congregation","company","mob"]
l_super_categ = get_coco_supercategories(coco)
k = 10
print(l_super_categ)


#l_img_id = find_picts_with_sing_and_crowd_person(coco,k=5)
#for img_id in l_img_id:
#    display_picture(coco,img_id)
#    print(get_caption(coco_caps,img_id))


data = count_sing_plur_crowd(coco)
plot_histo_number(data)


#print(l_img_id)
#for categ in l_super_categ:
    #print(categ)
    #print(refers_to_supercategory("apple", categ))
    #print(refers_to_supercategory("kid", categ))
    #print(refers_to_supercategory("crowd", categ))
    #l_img_id = get_k_random_image_ids_by_supercategory(coco, k)
    #l_sing,l_sev_sing,l_plur = starts_number(l_img_id)
    #print(np.mean(l_sing),np.mean(l_plur))











