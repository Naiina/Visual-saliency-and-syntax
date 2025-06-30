
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
import matplotlib.image as mpimg
from nltk.corpus import wordnet as wn
import pandas as pd
from depth_computation import load_midas_model
from collections import defaultdict
from tqdm import tqdm
from saliencyfilters import SaliencyFilters
from sys import argv
import torch
import cv2
import csv
import stanza
from utils import find_pos, get_synsets, intersection_hoi_and_loc,convert_loc_file_into_dict,prop_mentioned,rank_and_deprel_mentioned,get_dist_to_h
from utils import get_dict_categ_to_supercateg, get_annot_categ, get_normalised_size_and_pos_categ, mean_masked_saliency,hoi_interaction, get_coco_capt
import math
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import json
import stanza
#stanza.download('en')





np.random.seed(40)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def nb_noun_per_cap(loc_n_jsonl):

    l_nb_nouns_coco = []
    l_nb_nouns_loc_n = []

    d_loc_n_cations = convert_loc_file_into_dict(loc_n_jsonl)

    for id in tqdm(d_loc_n_cations.keys()):
        img = coco.loadImgs([int(id)])[0]
        l_captions_coco = get_coco_capt(img,coco_caps)
        for caption in l_captions_coco:
            tokens = word_tokenize(caption)
            tagged = pos_tag(tokens)
            tot_noun = sum([1 for (word, tag) in tagged if tag in ('NN', 'NNS')])
            l_nb_nouns_coco.append(tot_noun)

        l_captions_loc_n = d_loc_n_cations[id]
        for caption in l_captions_loc_n:
            tokens = word_tokenize(caption)
            tagged = pos_tag(tokens)
            tot_noun = sum([1 for (word, tag) in tagged if tag in ('NN', 'NNS')])
            l_nb_nouns_loc_n.append(tot_noun)
    return l_nb_nouns_coco,l_nb_nouns_loc_n


def compute_statistics(data, label):
    mean = np.mean(data)
    std = np.std(data)
    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    
    print(f"Statistics for {label}:")
    print(f"  Mean   : {mean}")
    print(f"  Std Dev: {std}")
    print(f"  Median : {median}")
    print(f"  Q1     : {q1}")
    print(f"  Q3     : {q3}")
    print()



def get_elem_and_hyperpath(categ,d_categ_synset):
    l_categ_syn = []
    if categ in d_categ_synset.keys():
        l_syn_name = d_categ_synset[categ]
        for elem in l_syn_name:
            syn = wn.synset(elem)
            hyp = syn.hypernym_paths() # warning: several hyperpath per syn
            l_categ_syn.append((syn,hyp))
    return l_categ_syn


def prop_mentionned_without_hoi_df(caption_file,d_categ_synset_and_exeptions,k,plot_boxes=False):
    
    l_categ = []
    l_area = []
    l_depth = []
    l_depth_rel = []
    l_depth_local_contrast = []
    l_super_categ = []
    l_color_sal = []
    l_color_rel = []
    l_color_local_contrast = []
    l_distance_to_center =  []
    l_mentioned = []
    l_id_pict = []
    l_distance_to_human = []

    with open(caption_file, 'r') as f:
        caption_syn = json.load(f)
    
    d_categ_to_supercateg = get_dict_categ_to_supercateg(coco)
    
    for it,id_img in enumerate(tqdm(caption_syn.keys())):
        if k>0 and it >k:
            break
        img = coco.loadImgs([int(id_img)])[0]
        img_path = os.path.join(imgDir, img['file_name'])
        img_array = io.imread(img_path)

        if len(img_array.shape) == 3 and img_array[0].shape[1] == 3 :
            if plot_boxes:
                annIds = coco.getAnnIds(imgIds=img['id'])
                anns = coco.loadAnns(annIds)
                img_path = os.path.join(imgDir, img['file_name'])
                I = cv2.imread(img_path)
                I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
                #cv2.rectangle(I, (int(float(x_pvic_min)), int(float(y_pvic_min))), (int(float(x_pvic_max)),int(float(y_pvic_max))), (0, 255, 0), 2)
                for ann in anns:
                    x, y, w, h = ann['bbox']
                    cv2.rectangle(I, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
                plt.imshow(I)
                plt.axis('off')
                plt.show()

            annot_categ_in_pict,l_anns = get_annot_categ(img,coco)
            annot_categ_in_pict = set(annot_categ_in_pict)
            print(annot_categ_in_pict,l_anns)
            for categ_name,annot in zip(annot_categ_in_pict,l_anns):

                l_syn_categ = d_categ_synset_and_exeptions[categ_name]["l_syn"] 
                l_categ_exeption = d_categ_synset_and_exeptions[categ_name]["exeptions"]
                l_caption_syn = caption_syn[id_img]
                prop_mention  = prop_mentioned(l_caption_syn,l_syn_categ,l_categ_exeption)
                area,nb,dist = get_normalised_size_and_pos_categ(img,categ_name,coco)
                print(anns)
                dist_to_h = get_dist_to_h(img,l_anns,anns)
                mean_depth_saliency,rel_depth_saliency,depth_local_contrsat = mean_masked_saliency(img,img_array,categ_name,coco,imgDir,metric="midas_depth",midas_transform=midas_transform,midas_model=midas_model,plot_fig=show_fig)
                color_saliency,rel_color_saliency,color_local_contrast = mean_masked_saliency(img,img_array,categ_name,coco,imgDir,metric="sf_saliency",sf =sf,plot_fig=show_fig)
                s_elem = d_categ_to_supercateg[categ_name]

                l_mentioned.append(prop_mention)
                l_area.append(area)
                l_depth.append(mean_depth_saliency)
                l_depth_rel.append(rel_depth_saliency)
                l_depth_local_contrast.append(depth_local_contrsat)
                l_color_sal.append(color_saliency)
                l_color_rel.append(rel_color_saliency)
                l_color_local_contrast.append(color_local_contrast)
                l_distance_to_center.append(dist)
                l_super_categ.append(s_elem)
                l_categ.append(categ_name)
                l_id_pict.append(id_img)
                
    d_all = {"Mentioned(%)":l_mentioned,"Size":l_area,"Distance to center":l_distance_to_center,"Depth":l_depth,"Relative depth":l_depth_rel,
             "Depth local contrast":l_depth_local_contrast,"Colour saliency":l_color_sal,"Relative Colour saliency":l_color_sal,
             "Colour local contrast":l_color_local_contrast,"Category":l_categ,"Super-category":l_super_categ}
    df = pd.DataFrame.from_dict(d_all)

    return df

def grammar_without_hoi_df(caption_file,d_categ_synset_and_exeptions,k,plot_boxes=False):

    l_categ = []
    l_area = []
    l_depth = []
    l_depth_rel = []
    l_depth_local_contrast = []
    l_super_categ = []
    l_color_sal = []
    l_color_rel = []
    l_color_local_contrast = []
    l_distance_to_center =  []
    l_rank_word_in_caption = []
    l_id_pict = []
    l_caption = []
    l_deprel_word_in_caption = []

    with open(caption_file, 'r') as f:
        caption_syn = json.load(f)
    d_categ_to_supercateg = get_dict_categ_to_supercateg(coco)
    
    for it,id_img in enumerate(tqdm(caption_syn.keys())):
        if k>0 and it >k:
            break
        img = coco.loadImgs([int(id_img)])[0]
        img_path = os.path.join(imgDir, img['file_name'])
        img_array = io.imread(img_path)

        if len(img_array.shape) == 3 and img_array[0].shape[1] == 3 :
            if plot_boxes:
                annIds = coco.getAnnIds(imgIds=img['id'])
                anns = coco.loadAnns(annIds)
                img_path = os.path.join(imgDir, img['file_name'])
                I = cv2.imread(img_path)
                I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
                for ann in anns:
                    x, y, w, h = ann['bbox']
                    cv2.rectangle(I, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
                plt.imshow(I)
                plt.axis('off')
                plt.show()

            annot_categ_in_pict,l_anns = get_annot_categ(img,coco)
            annot_categ_in_pict = set(annot_categ_in_pict)
            for categ_name in annot_categ_in_pict:
                l_syn_categ = d_categ_synset_and_exeptions[categ_name]["l_syn"] 
                l_categ_exeption = d_categ_synset_and_exeptions[categ_name]["exeptions"]
                l_caption_syn = caption_syn[id_img]
                
                area,nb,dist = get_normalised_size_and_pos_categ(img,categ_name,coco)
                mean_depth_saliency,rel_depth_saliency,depth_local_contrsat = mean_masked_saliency(img,img_array,categ_name,coco,imgDir,metric="midas_depth",midas_transform=midas_transform,midas_model=midas_model,plot_fig=show_fig)
                color_saliency,rel_color_saliency,color_local_contrast = mean_masked_saliency(img,img_array,categ_name,coco,imgDir,metric="sf_saliency",sf =sf,plot_fig=show_fig)
                s_elem = d_categ_to_supercateg[categ_name]
                l_5_rank,l_5_deprel = rank_and_deprel_mentioned(l_caption_syn,l_syn_categ,l_categ_exeption,nlp)
                for i,(rank,deprel) in enumerate(zip(l_5_rank,l_5_deprel)):
                    l_rank_word_in_caption.append(rank)
                    l_deprel_word_in_caption.append(deprel)
                    l_area.append(area)
                    l_depth.append(mean_depth_saliency)
                    l_depth_rel.append(rel_depth_saliency)
                    l_depth_local_contrast.append(depth_local_contrsat)
                    l_color_sal.append(color_saliency)
                    l_color_rel.append(rel_color_saliency)
                    l_color_local_contrast.append(color_local_contrast)
                    l_distance_to_center.append(dist)
                    l_super_categ.append(s_elem)
                    l_categ.append(categ_name)
                    l_id_pict.append(id_img)
                    l_caption.append(l_caption_syn[i])
                
    d_all = {"Rank":l_rank_word_in_caption,"Deprel":l_deprel_word_in_caption,"Size":l_area,"Distance to center":l_distance_to_center,"Depth":l_depth,"Relative depth":l_depth_rel,
             "Depth local contrast":l_depth_local_contrast,"Colour saliency":l_color_sal,"Relative Colour saliency":l_color_sal,
             "Colour local contrast":l_color_local_contrast,"Category":l_categ,"Super-category":l_super_categ,"img_id":l_id_pict,"caption":l_caption}
    df = pd.DataFrame.from_dict(d_all)


    return df






def prop_mentionned_with_hoi_df(hoi_csv_file,loc_n_jsonl,k,dataset_cap,plot_boxes=False,wsd=False,wsd_model=None,wsd_tok=None):

    cats = coco.loadCats(coco.getCatIds())
    l_categ_coco = [cat['name'] for cat in cats]
    d_categ_to_supercateg = get_dict_categ_to_supercateg(coco)
    catIds = coco.getCatIds(catNms="person")

    l_hoi = []
    l_prop_mentionned = []
    l_categ = []
    l_area = []
    l_pict_freq_categ = []
    d_pict_freq_categ = defaultdict(int)
    l_pict_freq_super_categ = []
    d_pict_freq_super_categ = defaultdict(int)
    l_depth = []
    l_depth_rel = []
    l_depth_local_contrast = []
    l_super_categ = []
    l_color_sal = []
    l_color_rel = []
    l_color_local_contrast = []
    l_distance_to_center =  []
    l_mentioned = []

    l_id_pict = []

    if dataset_cap == "localized_narratives":
        l_loc_and_hoi = list(intersection_hoi_and_loc(hoi_csv_file,loc_n_jsonl))
        d_loc_n_cations = convert_loc_file_into_dict(loc_n_jsonl)
    
    it = 0
    with open(hoi_csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in tqdm(reader):
            if k>0 and it >k:
                break
            it+=1
            id, x_pvic_min, y_pvic_min, x_pvic_max, y_pvic_max = row
            img = coco.loadImgs([int(id)])[0]
            img_path = os.path.join(imgDir, img['file_name'])
            img_array = io.imread(img_path)
            if len(img_array.shape) == 3 and img_array[0].shape[1] == 3 and id in l_loc_and_hoi:
                if plot_boxes:
                    annIds = coco.getAnnIds(imgIds=img['id'])
                    anns = coco.loadAnns(annIds)
                    img_path = os.path.join(imgDir, img['file_name'])
                    I = cv2.imread(img_path)
                    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
                    cv2.rectangle(I, (int(float(x_pvic_min)), int(float(y_pvic_min))), (int(float(x_pvic_max)),int(float(y_pvic_max))), (0, 255, 0), 2)
                    for ann in anns:
                        x, y, w, h = ann['bbox']
                        cv2.rectangle(I, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
                    plt.imshow(I)
                    plt.axis('off')
                    plt.show()

                d_elem_to_ment = defaultdict(list)
                annot_categ_in_pict,l_anns = get_annot_categ(img,coco)
                annot_categ_in_pict = set(annot_categ_in_pict)
                if dataset_cap == "coco":
                    l_captions = get_coco_capt(img,coco_caps)
                if dataset_cap == "localized_narratives":
                    l_captions = d_loc_n_cations[id]
                
                for caption in l_captions:
                    noun_in_anno_and_capt = []
                    synsets  = get_synsets(caption,wsd,wsd_tok,wsd_model,device)
                    print(caption)
                    print(synsets)
                    for syn in synsets:
                        name = syn.name()[:-5]
                        noun_in_anno_and_capt += find_pos(syn,annot_categ_in_pict)
                        
                    noun_in_anno_and_capt = set(noun_in_anno_and_capt)
                    for elem in list(annot_categ_in_pict):
                        if elem in noun_in_anno_and_capt:
                            d_elem_to_ment[elem].append(1)
                        else:
                            d_elem_to_ment[elem].append(0)
                for elem in list(annot_categ_in_pict):
                    area,nb,dist = get_normalised_size_and_pos_categ(img,elem,coco)
                    mean_depth_saliency,rel_depth_saliency,depth_local_contrsat = mean_masked_saliency(img,img_array,elem,coco,imgDir,metric="midas_depth",midas_transform=midas_transform,midas_model=midas_model,plot_fig=show_fig)
                    color_saliency,rel_color_saliency,color_local_contrast = mean_masked_saliency(img,img_array,elem,coco,imgDir,metric="sf_saliency",sf =sf,plot_fig=show_fig)
                    prop = d_elem_to_ment[elem]
                    s_elem = d_categ_to_supercateg[elem]
                    d_pict_freq_super_categ[s_elem]+=1
                    d_pict_freq_categ[elem]+=1
                    catIds = coco.getCatIds(catNms=elem)
                    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
                    anns = coco.loadAnns(annIds)
                    hoi = hoi_interaction(anns,x_pvic_min, y_pvic_min, x_pvic_max, y_pvic_max,img)
                    
                    for p in prop:
                        l_mentioned.append(p)
                        l_prop_mentionned.append(sum(prop)/len(prop))
                        l_area.append(area)
                        l_depth.append(mean_depth_saliency)
                        l_depth_rel.append(rel_depth_saliency)
                        l_depth_local_contrast.append(depth_local_contrsat)
                        l_color_sal.append(color_saliency)
                        l_color_rel.append(rel_color_saliency)
                        l_color_local_contrast.append(color_local_contrast)
                        l_distance_to_center.append(dist)
                        l_super_categ.append(s_elem)
                        l_categ.append(elem)
                        if hoi:
                            l_hoi.append(1)
                        else:
                            l_hoi.append(0)
                        l_id_pict.append(it)
                        
                
    for elem in l_categ:
        l_pict_freq_categ.append(d_pict_freq_categ[elem])
    for s_elem in l_super_categ:
        l_pict_freq_super_categ.append(d_pict_freq_super_categ[s_elem])
    d_all = {"mentionned(%)":l_prop_mentionned,"mentioned":l_mentioned,"HOI":l_hoi,"size":l_area,"distance_to_center":l_distance_to_center,"depth":l_depth,"depth_rel":l_depth_rel,"depth_local_contrast":l_depth_local_contrast,"color":l_color_sal,"color_rel":l_color_sal,"color_local_contrast":l_color_local_contrast,"categ":l_categ,"s_categ":l_super_categ,"categ freq":l_pict_freq_categ,"super categ freq":l_pict_freq_super_categ}
    df = pd.DataFrame.from_dict(d_all)
    d_vect = pd.get_dummies(df['s_categ']).astype(int).to_dict(orient='list')
    for k,v in d_vect.items():
        df[k]=v

    return df




def exctract_grammar_visual_features_with_hoi(hoi_csv_file,loc_n_jsonl,k,dataset_cap,plot_boxes=False):
    #stanza.download('en')       # This downloads the English models for the neural pipeline
    if device =="cuda":
        nlp = stanza.Pipeline('en',dir = "/disk/nfs/ostrom/s2523033/picture_saliency") # This sets up a default neural pipeline in English
    else:
        nlp = stanza.Pipeline('en')
    cats = coco.loadCats(coco.getCatIds())
    d_categ_to_supercateg = get_dict_categ_to_supercateg(coco)
    catIds = coco.getCatIds(catNms="person")
    #general 
    l_img_id = []
    l_categ = []
    l_super_categ = []

    # visual features
    l_hoi = []
    l_area = []
    l_depth = []
    l_color = []
    l_distance_to_center =  []
    l_pict_freq_categ = []
    d_pict_freq_categ = defaultdict(int)
    l_pict_freq_super_categ = []
    d_pict_freq_super_categ = defaultdict(int)
    
    #syntax features
    l_pos_in_caption = []
    l_mod = []
    l_rel = []
    l_deprel = []
    l_acl = []
    l_dependants = []
    l_rank = []

    it = 0
    with open(hoi_csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in tqdm(reader):
            if k>0 and it >k:
                break
            it+=1
            id, x_pvic_min, y_pvic_min, x_pvic_max, y_pvic_max = row
            img = coco.loadImgs([int(id)])[0]
            img_path = os.path.join(imgDir, img['file_name'])
            img_array = io.imread(img_path)
            if len(img_array.shape) == 3 and img_array[0].shape[1] == 3: # and id in l_loc_and_hoi:
                if plot_boxes:
                    annIds = coco.getAnnIds(imgIds=img['id'])
                    anns = coco.loadAnns(annIds)
                    img_path = os.path.join(imgDir, img['file_name'])
                    I = cv2.imread(img_path)
                    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
                    cv2.rectangle(I, (int(float(x_pvic_min)), int(float(y_pvic_min))), (int(float(x_pvic_max)),int(float(y_pvic_max))), (0, 255, 0), 2)
                    for ann in anns:
                        x, y, w, h = ann['bbox']
                        cv2.rectangle(I, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
                    plt.imshow(I)
                    plt.axis('off')
                    plt.show()

                d_elem_to_ment = defaultdict(list)
                annot_categ_in_pict,l_anns = get_annot_categ(img,coco)
                annot_categ_in_pict = set(annot_categ_in_pict)
                if dataset_cap == "coco":
                    l_captions = get_coco_capt(img,coco_caps)
                #if dataset_cap == "localized_narratives":
                #    l_captions = d_loc_n_cations[id]
                saved = {}
                
                for caption in l_captions:
                    d_deprel = get_deprel_data(caption,nlp)
                    l_syn = d_deprel["syn"]
                    added = []
                    for idx,syn in enumerate(l_syn): #loop over all nouns in caption
                        noun_in_anno_and_capt = find_pos(syn,annot_categ_in_pict)
                        for elem in set(noun_in_anno_and_capt):
                            if elem not in added:
                                added.append(elem)

                                l_pos_in_caption.append(d_deprel["idx_noun"][idx])
                                l_mod.append(d_deprel["mod"][idx])
                                l_rel.append(d_deprel["rel"][idx])
                                l_deprel.append(d_deprel["deprel"][idx])
                                l_dependants.append(d_deprel["dependants"][idx])
                                l_acl.append(d_deprel["acl"][idx])


                                if elem in saved.keys():
                                    area = saved[elem]["area"]
                                    mean_depth_saliency = saved[elem]["depth"]
                                    color_saliency = saved[elem]["color"]
                                    dist = saved[elem]["dist"]
                                    s_elem = saved[elem]["s_elem"]

                                else:
                                    area,nb,dist = get_normalised_size_and_pos_categ(img,elem,coco)
                                    mean_depth_saliency,rel_depth_saliency,depth_local_contrast = mean_masked_saliency(img,img_array,elem,coco,imgDir,metric="midas_depth",midas_transform=midas_transform,midas_model=midas_model,plot_fig=show_fig)
                                    color_saliency,rel_sal,color_local_contrast = mean_masked_saliency(img,img_array,elem,coco,imgDir,metric="sf_saliency",sf =sf,plot_fig=show_fig)
                                    prop = d_elem_to_ment[elem]
                                    s_elem = d_categ_to_supercateg[elem]
                                    d_pict_freq_super_categ[s_elem]+=1
                                    d_pict_freq_categ[elem]+=1
                                    catIds = coco.getCatIds(catNms=elem)
                                    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
                                    anns = coco.loadAnns(annIds)
                                    hoi = hoi_interaction(anns,x_pvic_min, y_pvic_min, x_pvic_max, y_pvic_max,img)
                                    saved[elem] = {"area":area,"depth":mean_depth_saliency,"color":color_saliency,"dist":dist,"s_elem":s_elem}

                                l_img_id.append(id)
                                l_area.append(area)
                                l_depth.append(mean_depth_saliency)
                                l_color.append(color_saliency)
                                l_distance_to_center.append(dist)
                                l_super_categ.append(s_elem)
                                l_categ.append(elem)
                                if hoi:
                                    l_hoi.append(1)
                                else:
                                    l_hoi.append(0)

                    
    for elem in l_categ:
        l_pict_freq_categ.append(d_pict_freq_categ[elem])
    for s_elem in l_super_categ:
        l_pict_freq_super_categ.append(d_pict_freq_super_categ[s_elem])
    d_all = {"HOI":l_hoi,"size":l_area,"distance_to_center":l_distance_to_center,"depth":l_depth,"color":l_color,"categ":l_categ,"s_categ":l_super_categ,"categ freq":l_pict_freq_categ,"super categ freq":l_pict_freq_super_categ,
             "pos_in_caption":l_pos_in_caption,"modifiers":l_mod,"relativisation":l_rel,"deprel":l_deprel,"acl":l_acl,"deprendants":l_dependants,"image_id":l_img_id}
    df = pd.DataFrame.from_dict(d_all)

    return df



def get_deprel_data(caption,nlp):
    doc = nlp(caption)
    l_idx_noun = []
    l_syn = []
    l_mod = []
    l_rel = []
    l_deprel = []
    l_acl = []
    l_dependants = []

    for sent in doc.sentences:
        #get nouns
        for word in sent.words:
            if word.upos == 'NOUN':
                syn = wn.synsets(word.lemma, 'n')
                if len(syn)>0:
                    l_syn.append(syn[0])
                    l_idx_noun.append(word.id)
                    
                    l_deprel.append(word.deprel)
                    l_mod.append(0)
                    l_rel.append(0)
                    l_acl.append(0)
                    l_dependants.append(0)
                
        # get amods and rel
        for i,n_idx in enumerate(l_idx_noun):
            for word in sent.words:
                if word.head == n_idx:
                    l_dependants[i]+=1
                    if "mod" in word.deprel:
                        l_mod[i]+=1
                    if word.deprel == 'acl:relcl':
                        l_rel[i]+=1
                    if "acl" in word.deprel:
                        l_acl[i]+=1
            
        d = {'idx_noun':l_idx_noun,'syn':l_syn,'deprel':l_deprel,"mod":l_mod,"rel":l_rel,"dependants":l_dependants,"acl":l_acl}
        return d





def get_relevant_pict(k,out_file,out_folder,categs=["person"]):

    cats = coco.loadCats(coco.getCatIds())
    l_categ_coco = [cat['name'] for cat in cats]

    catIds = coco.getCatIds(catNms=categs)
    imgIds = coco.getImgIds(catIds=catIds )
    if k>0:
        selected_imgIds = np.random.choice(imgIds, k, replace=False)
    else:
        selected_imgIds = imgIds

    l_img = coco.loadImgs(selected_imgIds.tolist())
    i = 0
    
    for img,id in tqdm(zip(l_img,selected_imgIds)):

        img_path = os.path.join(imgDir, img['file_name'])
        I = mpimg.imread(img_path)
        cv2.imwrite(out_folder+str(id)+'.jpg', I)
        i+=1

        if i%100 == 0:
            with open(out_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(selected_imgIds)


if __name__ == "__main__":

    save_file = '_deprel_rank_no_hoi_coco_output.csv'
    max_it = 10
    show_fig = False
    HOI_csv_file = "id_pvic_cleaned.csv"
    categ_syn_file = "categ_syn.json"
    wsd = True
    gramm = False
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pylab.rcParams['figure.figsize'] = (8.0, 10.0)
    dataDir='COCO'
    dataType='val2017'
    imgDir = os.path.join(dataDir, dataType)

    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
    annFile_cap = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
    coco=COCO(annFile)
    coco_caps=COCO(annFile_cap)

    sf = SaliencyFilters()
    midas_size = "small"
    midas_transform,midas_model = load_midas_model(midas_size)
    if gramm:
        if os.path.isdir("/disk/nfs/ostrom/s2523033/picture_saliency"):
            nlp = stanza.Pipeline('en',dir = "/disk/nfs/ostrom/s2523033/picture_saliency") # This sets up a default neural pipeline in English
            print("nlp on cluster")
        else:
            nlp = stanza.Pipeline('en')
            print("nlp local")

    with open("categ_to_synset.json", 'r') as f:
        d_categ_synset = json.load(f)


    #df = prop_mentionned_all_df(HOI_csv_file,localized_n_file,max_it,"localized_narratives",plot_boxes=False,wsd=False,pipe=None)
    
    dataset_cap = "localized_narratives"
    caption_file = dataset_cap+"_synsets.json"

    print("hey")
    df = prop_mentionned_without_hoi_df(caption_file,d_categ_synset,max_it,plot_boxes=False,)

    #df = grammar_without_hoi_df(caption_file,d_categ_synset,max_it,plot_boxes=False,)
    #print(df)
    #df.to_csv(dataset_cap+save_file, index=False)

    #dataset_cap = "coco"
    #caption_file = dataset_cap+"_synsets.json"
    #df = grammar_without_hoi_df(caption_file,d_categ_synset,max_it,plot_boxes=False,)
    #df.to_csv(dataset_cap+save_file, index=False)
    





