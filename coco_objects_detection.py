
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
from plots import plot_size_pos_line_regression, plot_metric_pos_bins, plot_metric_rank_bins, plot_hoi
from depth_computation import load_midas_model, get_depth
from collections import defaultdict
from tqdm import tqdm
from saliencyfilters import SaliencyFilters
from sys import argv
from skimage import io
import time
import torch
import cv2
import math
import csv
import seaborn as sns


np.random.seed(40)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



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
        #print(unique_vals)
        plt.imshow(mask)
    plt.show()
    return img['id']

def get_masks(coco,categ,img):
    l_masks = []
    catIds = coco.getCatIds(catNms=categ)  
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None) # list of annotation id. each segmentation has a specific one
    anns = coco.loadAnns(annIds) #list fo dict for each annot. keys:segmentation, area, iscrowd, 'image_id', 'bbox', 'category_id', 'id'(from annit_id)
    for ann in anns:
        l_masks.append(coco.annToMask(anns[0]))
    return l_masks
        

def get_normalised_size_and_pos_categ(img,categ,coco):
    #return sum of areas and min distance to center
    h,w = img["height"],img["width"]
    im_size = h*w
    catIds = coco.getCatIds(catNms=categ)
    print(img['id'])
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    print(annIds)
    anns = coco.loadAnns(annIds)
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



def mean_masked_saliency(img,elem,coco,imgDir,metric,sf =None,midas_transform=None,midas_model=None,plot_fig = False):
    l_mean_saliency = []
    l_masks = get_masks(coco,elem,img)
    img_path = os.path.join(imgDir, img['file_name'])
    img_array = io.imread(img_path)
    if metric == "sf_saliency":
        saliency = sf.compute_saliency(img_array) # array of ints between 0 and 1
    if metric == "midas_depth":
        saliency = get_depth(img_path,midas_transform,midas_model)

    for mask in l_masks:
        l_mean_saliency.append(saliency[mask == 1].mean())
    mean_saliency = sum(l_mean_saliency)/len(l_mean_saliency)
    rel_sal = mean_saliency/saliency.mean()
    blurred_image = cv2.GaussianBlur(img_array, (151, 151), 0)
    print(mean_saliency)
    if plot_fig and mean_saliency <0.2: 
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
        axes[2].set_title("mean depth: "+str(round(mean_saliency,3)), fontsize=40)
        plt.tight_layout()
        plt.show()

    return mean_saliency,rel_sal




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
    l_pict_freq = []
    l_depth = []
    l_depth_rel = []
    l_prop_mentionned = []
    l_categ = []
    l_color_sal = []
    l_distance_to_center =  []
    l_img = get_k_random_img(l_categ_coco,coco,k)
    d_categ_to_supercateg = get_dict_categ_to_supercateg(coco)
    
    for img in tqdm(l_img):
        d_elem_to_ment = defaultdict(list)
        annot_categ_in_pict = set(get_annot_categ(img,coco))
        l_captions = get_capt(img,coco_caps)
        #img_show(img,imgDir,coco,coco_caps,print_capt=True,contours=True)
        for caption in l_captions:
            noun_in_anno_and_capt = []
            d_w_syn,d_pos,d_order,d_pos_rel,d_order_rel  = identify_nouns_pos_and_synsets(caption)
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
            area,nb,dist = get_normalised_size_and_pos_categ(img,elem,coco)
            mean_depth_saliency,rel_depth_saliency = mean_masked_saliency(img,elem,coco,imgDir,metric="midas_depth",midas_transform=midas_transform,midas_model=midas_model)
            color_saliency,rel_sal = mean_masked_saliency(img,elem,coco,imgDir,metric="sf_saliency",sf =sf,plot_fig=show_fig)
            l_area.append(area)
            l_depth.append(mean_depth_saliency)
            l_depth_rel.append(rel_depth_saliency)
            l_color_sal.append(color_saliency)
            l_distance_to_center.append(dist)
            prop = d_elem_to_ment[elem]
            #l_prop_mentionned.append(f"{sum(prop)}/{len(prop)}")
            l_prop_mentionned.append(sum(prop)/len(prop))
            if super_categ:
                s_elem = d_categ_to_supercateg[elem]
                l_categ.append(s_elem)
            else:
                l_categ.append(elem)
    df = pd.DataFrame.from_dict({"mentionned(%)":l_prop_mentionned,"size":l_area,"distance_to_center":l_distance_to_center,"depth":l_depth,"depth_rel":l_depth_rel,"color":l_color_sal,"label":l_categ})

    return df
             


def prop_mentionned_HOI_df(csv_file,k,super_categ,plot_boxes=False):

    cats = coco.loadCats(coco.getCatIds())
    l_categ_coco = [cat['name'] for cat in cats]
    d_categ_to_supercateg = get_dict_categ_to_supercateg(coco)
    catIds = coco.getCatIds(catNms="person")

    l_hoi = []
    l_prop_mentionned = []
    l_categ = []
    
    it = 0
    with open(csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in tqdm(reader):
            if it >k:
                break
            it+=1
            id, x_pvic_min, y_pvic_min, x_pvic_max, y_pvic_max = row
            img = coco.loadImgs([int(id)])[0]
            
            #print("size",img["height"],img["width"])
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
            annot_categ_in_pict = set(get_annot_categ(img,coco))
            l_captions = get_capt(img,coco_caps)
            
            for caption in l_captions:
                noun_in_anno_and_capt = []
                d_w_syn,d_pos,d_order,d_pos_rel,d_order_rel  = identify_nouns_pos_and_synsets(caption)
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
                prop = d_elem_to_ment[elem]
                #print(elem)
                catIds = coco.getCatIds(catNms=elem)
                annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
                anns = coco.loadAnns(annIds)
                hoi = hoi_interaction(anns,x_pvic_min, y_pvic_min, x_pvic_max, y_pvic_max,img)
                if hoi:
                    l_hoi.append("HOI")
                else:
                    l_hoi.append("no HOI")
                l_prop_mentionned.append(sum(prop)/len(prop))
                if super_categ:
                    s_elem = d_categ_to_supercateg[elem]
                    l_categ.append(s_elem)
                else:
                    l_categ.append(elem)
    df = pd.DataFrame.from_dict({"mentionned(%)":l_prop_mentionned,"HOI":l_hoi,"label":l_categ})
    df.to_csv("hoi.csv", index=False)
    return df

def hoi_interaction(anns,x_pvic_min, y_pvic_min, x_pvic_max, y_pvic_max,img):
    #print("x_pv",x_coco_min, y_pvic_min, x_pvic_max, y_pvic_max)
    y_margin = img["height"]/50
    x_margin = img["width"]/50
    for ann in anns:
        bbox = ann["bbox"]
        #print("bbox",bbox)
        x_coco_min = bbox[0]
        y_coco_min = bbox[1]
        x_coco_max = bbox[0]+bbox[2]
        y_coco_max = bbox[1]+bbox[3]
        #print(type(x_coco_max),type(x_pvic_max))
        b_a = x_coco_min-x_margin < float(x_pvic_min) < x_coco_min+x_margin
        b_b =   x_coco_max-x_margin < float(x_pvic_max) < x_coco_max+x_margin
        b_c =   y_coco_min-y_margin < float(y_pvic_min) < y_coco_min+y_margin
        b_d =   y_coco_max-y_margin < float(y_pvic_max) < y_coco_max+y_margin
        if b_a and b_b and b_c and b_d:
            #print(bbox)
            return True
    return False
    
    


def exctract_df_pos_size(k,metric,rel_saliency=False,super_categ=True,sf=None ,midas_transform=None,midas_model=None,plot_fig=False):

    cats = coco.loadCats(coco.getCatIds())
    l_categ = [cat['name'] for cat in cats]

    l_pos_in_caption = []
    l_order_in_caption = []
    l_pos_rel_in_caption = []
    l_order_rel_in_caption = []
    l_area = []
    l_dist_to_center = []
    l_ment_categ = []
    l_num = []
    l_saliency = []
    l_rel_saliency = []

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
            area,nb,dist = get_normalised_size_and_pos_categ(img,elem,coco)
            mean_saliency,rel_saliency = mean_masked_saliency(img,elem,coco,imgDir,metric,sf ,midas_transform,midas_model,plot_fig = plot_fig)
            if rel_saliency:
                saliency = rel_saliency
            else:
                saliency = mean_saliency

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
            l_dist_to_center.append(dist)
            l_saliency.append(mean_saliency)
            l_rel_saliency.append(rel_saliency)
            l_num.append(nb)
            if super_categ:
                s_elem = d_categ_to_supercateg[elem]
                l_ment_categ.append(s_elem)
            else:
                l_ment_categ.append(elem)
    df = pd.DataFrame.from_dict({"position":l_pos_in_caption,"rank":l_order_in_caption,"position_rel":l_pos_rel_in_caption,"rank_rel":l_order_rel_in_caption,"size":l_area,"distance_to_center":l_dist_to_center,"saliency":l_saliency,"saliency_rel":l_rel_saliency,"label":l_ment_categ,"number":l_num})
    return df





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
        #    l_categ = set(get_annot_categ(img,coco))
        #    img_show(img,imgDir,coco,coco_caps,print_capt=False,contours=True)

        img_path = os.path.join(imgDir, img['file_name'])
        I = mpimg.imread(img_path)
        cv2.imwrite(out_folder+str(id)+'.jpg', I)
        i+=1

        if i%100 == 0:
            with open(out_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(selected_imgIds)


if __name__ == "__main__":
    
    extract = False
    plot = True
    save_file = '5000_coco_output_prop_mentionned.csv'
    avg = False
    nb_pict = 5000
    super_categ = True
    sf_saliency = True
    midas_depth = True
    midas_size = "small"
    metric = "sf_saliency" #"midas_depth" "sf_saliency"
    show_fig = True
    HOI_csv_file = "id_pvic_boxes.csv"


    

    if extract:
        pylab.rcParams['figure.figsize'] = (8.0, 10.0)
        dataDir='COCO'
        dataType='val2017'
        imgDir = os.path.join(dataDir, dataType)

        annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
        annFile_cap = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
        coco=COCO(annFile)
        coco_caps=COCO(annFile_cap)

        #get_relevant_pict(1000,"person_pict/pict_id.csv","person_pict/",categs=["person"])
        #exit()

        if sf_saliency:
            sf = SaliencyFilters()
        else:
            sf = None
        if midas_depth:
            midas_transform,midas_model = load_midas_model(midas_size)
        else:
            midas_transform = None
            midas_model = None

        #df = prop_mentionned_size_df(5000,True)

        
        #df = exctract_df_pos_size(nb_pict,metric,rel_saliency=False,super_categ=True,sf=sf,midas_transform=midas_transform,midas_model=midas_model,plot_fig=show_fig)
        df = prop_mentionned_HOI_df(HOI_csv_file,nb_pict,True)
        #df.to_csv(save_file, index=False)
        #print(df)
        #exit()

        if avg:
            avg_df = df.groupby(['label', 'number'])[['size', 'position']].mean().reset_index()
            df.to_csv("avg_"+save_file, index=False)
        
    if plot:
        #plot_hoi("hoi.csv")
        #barplot(save_file)
        #plot_size_prop(save_file)
        plot_heatmap_prop_mentionned(save_file,"size")
        #plot_size_pos_line_regression(save_file)
        #rel = False
        #plot_metric_rank_bins(save_file,rel=rel,metric="saliency_rel")
        #plot_metric_pos_bins(save_file,rel=rel,metric="saliency_rel")









