

from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import defaultdict
import pandas as pd
#from wsd_without_pipe import wsd_llm
from nltk.tokenize import MWETokenizer
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
from nltk.corpus import wordnet as wn
import pandas as pd
from depth_computation import get_depth
import cv2
import csv
import math
import json






def identify_nouns_pos_and_synsets(sentence,wsd=False,pipe=None):
   
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
                #syn = wsd_llm(word, sentence,pipe)
                syn = None
            else:
                syn = wn.synsets(word, 'n')[0]
            print(word,tag,syn,syn.definition())
            if syn not in noun_pos.keys(): #consider only the first occurence of a synset
                noun_synsets[word] = syn
                noun_pos[syn]=(i+1)
                noun_order[syn]=order
                noun_pos_rel[syn]=(i+1)/len_sent
                noun_order_rel[syn]=order/tot_noun
    
    return noun_synsets, noun_pos, noun_order, noun_pos_rel,noun_order_rel


def get_synsets(sentence,wsd=False,tok=None,model=None,device="cuda"):

    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    
    synsets = []

    for i,(word, tag) in enumerate(tagged):
        if tag in ('NN', 'NNS'):
            if wsd:
                #syn = wsd_llm(word, sentence,tok,model,device)
                syn = None
                synsets.append(syn)
            else:
                
                l_syn = wn.synsets(word, 'n')
                if len(l_syn)>0:
                    synsets.append(l_syn[0])
        
    return list(set(synsets))

#mwe_tokenizer = MWETokenizer([('hot', 'dog'), ('ice', 'cream')], separator='_')





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




def print_coco_categ(coco):
    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n{}'.format(' '.join(nms)))


def get_k_random_img(categ,coco,k=1):
    catIds = coco.getCatIds(catNms=[categ])
    imgIds = coco.getImgIds(catIds=catIds )
    if k>0:
        selected_imgIds = np.random.choice(imgIds, k, replace=False)
    else:
        selected_imgIds = imgIds
    imgs = coco.loadImgs(selected_imgIds.tolist())
    return imgs

def get_coco_capt(img,coco_caps):
    annIds = coco_caps.getAnnIds(imgIds=img['id'])
    anns = coco_caps.loadAnns(annIds)
    l_captions = [d["caption"] for d in anns]
    return l_captions

def img_show(img,imgDir,coco,coco_caps,contours=False,plt_mask=False,print_capt=True):
    img_path = os.path.join(imgDir, img['file_name'])
    I = mpimg.imread(img_path)
    if print_capt:
        print("id",img['id'])
        annIds = coco_caps.getAnnIds(imgIds=img['id'])
        anns = coco_caps.loadAnns(annIds)
        coco_caps.showAnns(anns)
    plt.imshow(I); plt.axis('off')
    if contours:
        catIds = coco.getCatIds()
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        coco.showAnns(anns)
        
    if plt_mask:
        catIds = coco.getCatIds() # list of int, categ id (1: person 18:chien ....) 
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None) # list of annotation id. each segmentation has a specific one
        anns = coco.loadAnns(annIds) #list fo dict for each annot. keys:segmentation, area, iscrowd, 'image_id', 'bbox', 'category_id', 'id'(from annit_id)
        mask = coco.annToMask(anns[0])

        for i in range(len(anns)):
            mask += coco.annToMask(anns[i])
        unique_vals = np.unique(mask)
        plt.imshow(mask)
    plt.show()
    return img['id']

def get_masks_and_box(coco,categ,img):
    l_masks = []
    l_box = []
    catIds = coco.getCatIds(catNms=categ)  
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None) # list of annotation id. each segmentation has a specific one
    anns = coco.loadAnns(annIds) #list fo dict for each annot. keys:segmentation, area, iscrowd, 'image_id', 'bbox', 'category_id', 'id'(from annot_id)
    
    for ann in anns:
        l_masks.append(coco.annToMask(anns[0]))
        l_box.append(ann["bbox"])

    return l_masks,l_box
        

def get_normalised_size_and_pos_categ(img,categ,coco,anns):
    #return sum of areas and min distance to center
    h,w = img["height"],img["width"]
    im_size = h*w
    #catIds = coco.getCatIds(catNms=categ)
    #annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    #anns = coco.loadAnns(annIds)
    area = 0
    nb = 0
    is_crowd = 0
    dist = 1
    for ann in anns:
        area+=ann["area"]
        nb+=1
        is_crowd+=ann["iscrowd"]
        bbox = ann["bbox"]
        d = normalized_center_distance(bbox, (w,h))
        dist = min(dist,d)
    n_area = area/im_size
    if is_crowd>0:
        return n_area , "crowd", dist
    else:
        if nb ==1:
            return n_area, "one", dist
        else:
            return n_area, "several", dist


def normalized_center_distance(bbox, image_size):
    x_min, y_min, x_len, y_len = bbox
    image_width, image_height = image_size
    bbox_center_x = x_min + x_len / 2
    bbox_center_y = y_min + y_len / 2
    image_center_x = image_width / 2
    image_center_y = image_height / 2
    distance = math.sqrt((bbox_center_x - image_center_x)**2 + (bbox_center_y - image_center_y)**2)
    image_diagonal = math.sqrt(image_width**2 + image_height**2)
    normalized_distance = distance / image_diagonal
    return normalized_distance


def get_dist_to_h(img,l_anns,anns):
    print(anns)
    exit()
    x_min, y_min, x_len, y_len = 1,2,3,4
    image_width, image_height = 1,2
    bbox_center_x = x_min + x_len / 2
    bbox_center_y = y_min + y_len / 2
    image_center_x = image_width / 2
    image_center_y = image_height / 2
    distance = math.sqrt((bbox_center_x - image_center_x)**2 + (bbox_center_y - image_center_y)**2)
    image_diagonal = math.sqrt(image_width**2 + image_height**2)
    normalized_distance = distance / image_diagonal
    return normalized_distance

def get_human_box(coco,img):
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=[1], iscrowd=0)
    anns = coco.loadAnns(annIds)
    for ann in anns:
        print(ann['bbox'])
    exit()


def mean_masked_saliency(img,img_array,elem,coco,imgDir,metric,sf =None,midas_transform=None,midas_model=None,plot_fig = False):
    l_mask_saliency = []
    #l_box_saliency = []
    l_loc_sal = []
    l_masks,l_bbox = get_masks_and_box(coco,elem,img) #x_min, y_min, x_len, y_len = bbox
    img_path = os.path.join(imgDir, img['file_name'])
    
    if metric in "sf_saliency":
        saliency = sf.compute_saliency(img_array) # array of ints between 0 and 1
    if metric == "midas_depth":
        saliency = get_depth(img_path,midas_transform,midas_model)

    for mask,bbox in zip(l_masks,l_bbox):
        mask_sal = saliency[mask == 1].mean()
        l_mask_saliency.append(mask_sal)
        x_min, y_min, x_len, y_len = bbox
        contour_mask = np.zeros_like(mask, dtype=np.uint8)
        contour_mask[int(y_min):int(y_min)+int(y_len),int(x_min):int(x_min)+int(x_len)]=1
        contour_mask[mask == 1] = 0
        #l_box_saliency.append(saliency[contour_mask == 1].mean())
        loc_sal = np.array([abs(mask_sal-i) for i in saliency[contour_mask == 1]]).mean()
        l_loc_sal.append(loc_sal)
    local_contrast = np.array(l_loc_sal).mean()
    mask_saliency = sum(l_mask_saliency)/len(l_mask_saliency)
    rel_sal = mask_saliency/saliency.mean()
    
    
    blurred_image = cv2.GaussianBlur(img_array, (151, 151), 0)
    if plot_fig: 
        #saved_pict +=1
        fig, axes = plt.subplots(1, 4, figsize=(15, 5))
        axes[0].imshow(img_array)
        axes[0].axis('off')  
        axes[1].imshow(l_masks[0],cmap='gray')
        axes[1].axis('off')
        axes[2].imshow(saliency,cmap='gray')
        axes[2].axis('off')
        axes[3].imshow(blurred_image)
        axes[3].axis('off')
        axes[1].set_title(elem, fontsize=40)
        axes[2].set_title("mean depth: "+str(round(mask_saliency,3))+"_"+str(round(local_contrast,3)), fontsize=40)
        plt.tight_layout()
        #plt.show()
        plt.savefig("big_depth"+str(saved_pict)+".pdf",format="pdf")

    return mask_saliency,rel_sal, local_contrast




def get_dict_id_to_name(coco):
    cats = coco.loadCats(coco.getCatIds())
    return {cat["id"]:cat["name"] for cat in cats}




def get_dict_supercateg_to_categ(coco):
    cats = coco.loadCats(coco.getCatIds())
    d = defaultdict(list)
    for cat in cats:
        d[cat["supercategory"]].append(cat["name"])
    return d

def get_dict_categ_to_supercateg(coco):
    cats = coco.loadCats(coco.getCatIds())
    return {cat["name"]:cat["supercategory"] for cat in cats}


def get_annot_categ(img,coco):
    id_to_name = get_dict_id_to_name(coco)
    catIds = coco.getCatIds()
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    l_categ = []
    l_anns = []
    for ann in anns:
        cat_id = ann["category_id"]
        categ_name = id_to_name[cat_id]
        l_categ.append(categ_name)
        l_anns.append(ann)
    return l_categ,l_anns

             
def hoi_interaction(anns,x_pvic_min, y_pvic_min, x_pvic_max, y_pvic_max,img):
    y_margin = img["height"]/50
    x_margin = img["width"]/50
    for ann in anns:
        bbox = ann["bbox"]
        x_coco_min = bbox[0]
        y_coco_min = bbox[1]
        x_coco_max = bbox[0]+bbox[2]
        y_coco_max = bbox[1]+bbox[3]
        b_a = x_coco_min-x_margin < float(x_pvic_min) < x_coco_min+x_margin
        b_b =   x_coco_max-x_margin < float(x_pvic_max) < x_coco_max+x_margin
        b_c =   y_coco_min-y_margin < float(y_pvic_min) < y_coco_min+y_margin
        b_d =   y_coco_max-y_margin < float(y_pvic_max) < y_coco_max+y_margin
        if b_a and b_b and b_c and b_d:
            return True
    return False




def intersection_hoi_and_loc(hoi_csv,loc_n_jsonl):
    l_hoi_id = []
    #d=convert_loc_file_into_dict(loc_n_jsonl).keys()

    l_loc_id = list(convert_loc_file_into_dict(loc_n_jsonl).keys())

    with open(hoi_csv, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            id, x_pvic_min, y_pvic_min, x_pvic_max, y_pvic_max = row
            l_hoi_id.append(id)
    print("hoi",len(l_hoi_id))
    print("loc",len(l_loc_id))
    l_intersection = set(l_loc_id)&set(l_hoi_id)
    print(len(l_intersection))
    return l_intersection


def convert_loc_file_into_dict(loc_n_jsonl):
    d = defaultdict(list)
    with open(loc_n_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            d[data["image_id"]].append(data["caption"]) 
    return d


def categ_is_mentioned(syn_categ,l_hyp_categ,l_syn_caption):
    for syn_caption_name in l_syn_caption:
        syn_caption = wn.synset(syn_caption_name)
        l_hyp_caption = syn_caption.hypernyms()
        #categ is in hyperpath of an elem in the caption
        for hyp_caption in l_hyp_caption:
            if syn_categ in hyp_caption:
                return True
        #
        for hyp_categ in l_hyp_categ:
            if syn_caption in hyp_categ:
                return True
        print("deal with special case")
        return False
    

def get_hyp_and_syn_set(l_syn_name):
    l_hyp = []
    l_syn = []
    for syn_name in l_syn_name:
        syn = wn.synset(syn_name)
        l_syn.append(syn)
        for hyp_path in syn.hypernym_paths():
          for hyp in hyp_path:
              l_hyp.append(hyp)

    return set(l_syn), set(l_hyp)




def prop_mentioned(l_5_caption_syn,l_syn_categ,l_categ_exption):
    #l_caption_syn: list of [caption, l_nouns, l_syn_names]
    #l_categ_syn: list of syn NB: on categ can have several corresponding synsets
    #l_categ_exeption: catch two words categ which are not recognised by wordnet
    is_ment = []
    
    for caption,l_nouns,l_syn_caption in l_5_caption_syn:
        set_syn_caption,set_hyp_caption = get_hyp_and_syn_set(l_syn_caption)
        set_syn_categ,set_hyp_categ = get_hyp_and_syn_set(l_syn_categ)
        if set_syn_caption & set_hyp_categ:
            is_ment.append(1)
        elif set_syn_categ & set_hyp_caption:
            is_ment.append(1)
        elif set(l_categ_exption) & set(l_nouns):
            is_ment.append(1)
        else:
            is_ment.append(0)
    return np.mean(np.array(is_ment))
        

def get_deprel(caption,noun,nlp):
    doc = nlp(caption)
    for i, sent in enumerate(doc.sentences):
        for token in sent.words:
            if token.text == noun:
                head_n = token.head
                deprel = token.deprel
                if deprel == "conj":
                    for dep in sent.words:      
                       if dep.id == head_n:
                           deprel = dep.deprel
                           head_n = dep.head
                           break
                if deprel == "nsubj":
                    l_dep_deprel = [dep.deprel for dep in sent.words if dep.head == head_n]
                    if "obj" in l_dep_deprel:
                        return "nsubj:t"
                    else:
                        return "nsubj:i"
                    
                return deprel



def rank_and_deprel_mentioned(l_5_caption_syn,l_syn_categ,l_categ_exption,nlp):
    #l_caption_syn: list of [caption, l_nouns, l_syn_names]
    #l_categ_syn: list of syn NB: on categ can have several corresponding synsets
    #l_categ_exeption: catch two words categ which are not recognised by wordnet

    #subj:i and subj:t for transitive and intransitive subjects of active sentences. For conj, we consider the deprel of the head of the conj
    l_rank = []
    l_deprel = []
    
    for caption,l_nouns,l_syn_caption in l_5_caption_syn:
        mem_rank = -1
        mem_deprel = "none"
        filtered = [(noun, syn) for noun, syn in zip(l_nouns, l_syn_caption) if syn != "picture.n.01"]
        l_nouns, l_syn_caption = zip(*filtered) if filtered else ([], [])
        l_nouns = list(l_nouns)
        l_syn_caption = list(l_syn_caption)
        for rank,(noun,noun_syn) in enumerate(zip(l_nouns,l_syn_caption)):
            
            set_syn_caption,set_hyp_caption = get_hyp_and_syn_set([noun_syn])
            set_syn_categ,set_hyp_categ = get_hyp_and_syn_set(l_syn_categ)
            if set_syn_caption & set_hyp_categ or set_syn_categ & set_hyp_caption or set(l_categ_exption) & set(l_nouns):
                mem_rank = rank
                mem_deprel = get_deprel(caption,noun,nlp)
                break
        l_rank.append(mem_rank)
        l_deprel.append(mem_deprel)
        
    return l_rank,l_deprel



    

