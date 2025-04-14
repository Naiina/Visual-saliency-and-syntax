
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
import matplotlib.image as mpimg
import pycocotools._mask as _mask
from PIL import Image
from utils import identify_nouns_pos_and_synsets, find_pos
from nltk.corpus import wordnet as wn
import pandas as pd
from plots import plot_size_pos, plot_size_pos_nb,barplot, plot_size_prop, plot_heatmap_prop_mentionned
from plots import plot_size_pos_line_regression, plot_size_pos_bins, plot_size_rank_bins
from collections import defaultdict
from tqdm import tqdm
from saliencyfilters import SaliencyFilters
from sys import argv
from skimage import io
import time


np.random.seed(40)




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

def get_capt(img,coco_caps):
    annIds = coco_caps.getAnnIds(imgIds=img['id'])
    anns = coco_caps.loadAnns(annIds)
    l_captions = [d["caption"] for d in anns]
    return l_captions

def img_show(img,imgDir,coco,coco_caps,contours=False,plt_mask=False,print_capt=True):
    img_path = os.path.join(imgDir, img['file_name'])
    I = mpimg.imread(img_path)
    if print_capt:
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
        print(unique_vals)
        plt.imshow(mask)
    plt.show()

def get_masks(coco,categ,img):
    l_masks = []
    catIds = coco.getCatIds(catNms=categ)  
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None) # list of annotation id. each segmentation has a specific one
    anns = coco.loadAnns(annIds) #list fo dict for each annot. keys:segmentation, area, iscrowd, 'image_id', 'bbox', 'category_id', 'id'(from annit_id)
    for ann in anns:
        l_masks.append(coco.annToMask(anns[0]))
    return l_masks
        

def get_normalised_size_categ(img,categ,coco):
    im_size = img["height"]*img["width"]
    catIds = coco.getCatIds(catNms=categ)
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    area = 0
    nb = 0
    is_crowd = 0
    for ann in anns:
        
        area+=ann["area"]
        nb+=1
        is_crowd+=ann["iscrowd"]
    n_area = area/im_size
    if is_crowd>0:
        return n_area , "crowd"
    else:
        if nb ==1:
            return n_area, "one"
        else:
            return n_area, "several"
        
def color_saliency(img,elem,coco,sf,imgDir):
    l_mean_saliency = []
    l_masks = get_masks(coco,elem,img)
    img_path = os.path.join(imgDir, img['file_name'])
    img_array = io.imread(img_path)
    saliency = sf.compute_saliency(img_array) # array of ints between 0 and 1

    for mask in l_masks:
        l_mean_saliency.append(saliency[mask == 1].mean())
    mean_saliency = sum(l_mean_saliency)/len(l_mean_saliency)
    rel_sal = mean_saliency/saliency.mean()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_array)
    axes[0].axis('off')  
    axes[1].imshow(l_masks[0],cmap='gray')
    axes[1].axis('off')
    axes[2].imshow(saliency,cmap='gray')
    axes[2].axis('off')
    axes[1].set_title(elem, fontsize=40)
    axes[2].set_title("saliency: "+str(round(rel_sal,3)), fontsize=40)
    plt.tight_layout()
    plt.show()

    return rel_sal




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
    for ann in anns:
        cat_id = ann["category_id"]
        categ_name = id_to_name[cat_id]
        l_categ.append(categ_name)
    return l_categ


def prop_mentionned_size_df(k,super_categ):

    cats = coco.loadCats(coco.getCatIds())
    l_categ_coco = [cat['name'] for cat in cats]

    l_area = []
    l_prop_mentionned = []
    l_categ = []

    l_img = get_k_random_img(l_categ_coco,coco,k)
    d_categ_to_supercateg = get_dict_categ_to_supercateg(coco)
    
    for img in tqdm(l_img):
        d_elem_to_ment = defaultdict(list)
        annot_categ_in_pict = set(get_annot_categ(img,coco))
        l_captions = get_capt(img,coco_caps)
        #img_show(img,imgDir,coco,coco_caps,print_capt=True,contours=True)
        for caption in l_captions:
            noun_in_anno_and_capt = []
            d_w_syn,d_pos,d_order = identify_nouns_pos_and_synsets(caption)
            for syn in d_pos.keys():
                name = syn.name()[:-5]
                noun_in_anno_and_capt += find_pos(syn,annot_categ_in_pict)
            noun_in_anno_and_capt = set(noun_in_anno_and_capt)
            for elem in list(annot_categ_in_pict):
                if elem in noun_in_anno_and_capt:
                    d_elem_to_ment[elem].append(1)
                else:
                    d_elem_to_ment[elem].append(0)
        for elem in list(annot_categ_in_pict):
            area,nb = get_normalised_size_categ(img,elem,coco)
            l_area.append(area)
            prop = d_elem_to_ment[elem]
            #l_prop_mentionned.append(f"{sum(prop)}/{len(prop)}")
            l_prop_mentionned.append(sum(prop)/len(prop))
            if super_categ:
                s_elem = d_categ_to_supercateg[elem]
                l_categ.append(s_elem)
            else:
                l_categ.append(elem)
    df = pd.DataFrame.from_dict({"mentionned(%)":l_prop_mentionned,"size":l_area,"label":l_categ})

    return df
             


def exctract_df_pos_size(k,super_categ):

    cats = coco.loadCats(coco.getCatIds())
    l_categ = [cat['name'] for cat in cats]

    l_pos_in_caption = []
    l_order_in_caption = []
    l_pos_rel_in_caption = []
    l_order_rel_in_caption = []
    l_area = []
    l_ment_categ = []
    l_num = []
    l_color_saliency = []

    l_img = get_k_random_img(l_categ,coco,k)
    d_categ_to_supercateg = get_dict_categ_to_supercateg(coco)
    #print(get_dict_supercateg_to_categ(coco))
    

    for img in tqdm(l_img):
        l_captions = get_capt(img,coco_caps)
        #img_show(img,imgDir,coco,coco_caps,print_capt=True,contours=True)
        d_elem_to_pos = defaultdict(list)
        d_elem_to_order = defaultdict(list)
        d_elem_to_pos_rel = defaultdict(list)
        d_elem_to_order_rel = defaultdict(list)
        annot_categ_in_pict = set(get_annot_categ(img,coco))
        for caption in l_captions:
            d_w_syn,d_pos,d_order,d_pos_rel,d_order_rel = identify_nouns_pos_and_synsets(caption)
            for syn,pos in d_pos.items():
                noun_in_anno_and_capt = find_pos(syn,annot_categ_in_pict)
                for elem in noun_in_anno_and_capt:
                    d_elem_to_pos[elem].append(pos)
                    order = d_order[syn]
                    d_elem_to_order[elem].append(order)
                    order_rel = d_order_rel[syn]
                    d_elem_to_order_rel[elem].append(order_rel)
                    pos_rel = d_pos_rel[syn]
                    d_elem_to_pos_rel[elem].append(pos_rel)
        for elem,l_pos in d_elem_to_pos.items():
            area,nb = get_normalised_size_categ(img,elem,coco)
            c_saliency = color_saliency(img,elem,coco,sf,imgDir)
            print(elem,c_saliency)

            pos = sum(l_pos) / len(l_pos)
            l_pos_rel = d_elem_to_pos_rel[elem]
            pos_rel = sum(l_pos_rel)/len(l_pos_rel)

            l_order = d_elem_to_order[elem]
            order = sum(l_order)/len(l_order)
            l_order_rel = d_elem_to_order_rel[elem]
            order_rel = sum(l_order_rel)/len(l_order_rel)
            
            l_pos_in_caption.append(pos)
            l_order_in_caption.append(order)
            l_pos_rel_in_caption.append(pos_rel)
            l_order_rel_in_caption.append(order_rel)
            l_area.append(area)
            l_color_saliency.append(c_saliency)
            l_num.append(nb)
            if super_categ:
                s_elem = d_categ_to_supercateg[elem]
                l_ment_categ.append(s_elem)
            else:
                l_ment_categ.append(elem)
    df = pd.DataFrame.from_dict({"position":l_pos_in_caption,"rank":l_order_in_caption,"position_rel":l_pos_rel_in_caption,"rank_rel":l_order_rel_in_caption,"size":l_area,"color_saliency":l_color_saliency,"label":l_ment_categ,"number":l_num})
    return df




if __name__ == "__main__":
    
    extract = True
    plot = False
    save_file = '5_coco_output_rank_pos_size_saliency.csv'
    avg = False
    nb_pict = 5
    super_categ = True
    color_sal = True

    if extract:
        pylab.rcParams['figure.figsize'] = (8.0, 10.0)
        dataDir='COCO'
        dataType='val2017'
        imgDir = os.path.join(dataDir, dataType)

        annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
        annFile_cap = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
        coco=COCO(annFile)
        coco_caps=COCO(annFile_cap)

        if color_sal:
            sf = SaliencyFilters()

        
        df = exctract_df_pos_size(nb_pict,super_categ=True)
        print(df)
        exit()
        #df = prop_mentionned_size_df(nb_pict,super_categ)
        
        df.to_csv(save_file, index=False)

        if avg:
            avg_df = df.groupby(['label', 'number'])[['size', 'position']].mean().reset_index()
            df.to_csv("avg_"+save_file, index=False)
        
    if plot:
        #barplot(save_file)
        #plot_size_prop(save_file)
        #plot_heatmap_prop_mentionned(df)
        #plot_size_pos_line_regression(save_file)
        rel = True
        plot_size_rank_bins(save_file,rel = rel)
        plot_size_pos_bins(save_file,rel=rel)
        rel = False
        plot_size_rank_bins(save_file,rel = rel)
        plot_size_pos_bins(save_file,rel=rel)









