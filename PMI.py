import pandas as pd
import os
from collections import defaultdict
import ast
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
from tqdm import tqdm

#Rank,Deprel,Size,Distance to center

def discretisation(feat_value,feat_min,feat_max,nb_bins):
    
    bin_size = (feat_max - feat_min)/nb_bins
    if feat_value >= feat_max:
        return (feat_max-bin_size)
    idx = int((feat_value - feat_min) // bin_size)
    bin_mean_value = feat_min + idx * bin_size + bin_size/2
    return bin_mean_value
    
def discretisation_log(feat_value, feat_min, feat_max, nb_bins):
    esp = 0.02
    if feat_value+esp <= 0:
        raise ValueError("Log discretisation requires feat_value > 0")
    #print(feat_min+esp,feat_max)
    # Create log-spaced edges
    log_edges = np.logspace(np.log10(feat_min+esp), np.log10(feat_max), nb_bins+1)
    #print(log_edges)
    # If value is above max, assign to last bin
    if feat_value+esp >= feat_max:
        return (log_edges[-2] + log_edges[-1]) / 2
    
    # Find bin index
    idx = np.digitize(feat_value+esp, log_edges) - 1
    
    # Bin midpoint (in linear space, not log space)
    bin_mean_value = (log_edges[idx] + log_edges[idx+1]) / 2
    return bin_mean_value


def get_feat_gram_dict(s_feat,g_feat,nb_bins,file_path,categ = "all"):
    # sal_feat: saliency feat, such as size, depth ...
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    if categ != "all":
        df = df[df["Super-category"]==categ]
    d_sal_gram = defaultdict(lambda: defaultdict(int))
    if s_feat in ["Rank","Relative depth","Distance to center","Size"]:
        try:
            sal_min_val = df[s_feat].min()
            sal_max_val = df[s_feat].max()
        except KeyError:
            print(f"Column '{s_feat}' not found in DataFrame.")
    #print(sal_max_val,sal_min_val)
    for index, row in df.iterrows():
        if int(row["Rank"])!= -1: # is mentioned
            
            g_feat_v = row[g_feat]
            if s_feat in ["Rank","Distance to center","Relative depth","Size"]:
                s_feat_v = row[s_feat]
                if s_feat in ["Size","Relative depth"]:
                    s_feat_v =  discretisation_log(float(s_feat_v),sal_min_val,sal_max_val,nb_bins)
                else:
                    s_feat_v =  discretisation(float(s_feat_v),sal_min_val,sal_max_val,nb_bins)
                d_sal_gram[s_feat_v][g_feat_v]+=1


    print(dict(d_sal_gram))
    #exit()
    return d_sal_gram



def plot_voice(s_feat,nb_bins,file_path,categ = "all"):
 
    d_count = get_feat_gram_dict(s_feat,"Deprel",nb_bins,file_path,categ)
    df = pd.DataFrame.from_dict(d_count, orient='index')
    existing_cols = [c for c in ["obl:agent","obj","nsubj:t","nsubj:i","nsubj:pass"] if c in df.columns]

    df = df[existing_cols]
    df = df.fillna(0)
    
    grand_total = df.values.sum()

    # Calculate proportions
    df = df / grand_total
    
    # Reset index to get it in long form
    df = df.reset_index().melt(id_vars='index', var_name="Deprel", value_name='Proportion')
    #df = df[df["Proportion"] >= 10]
    #df["Proportion"] = df["Proportion"]/grand_total
    #print(df)
    #exit()
    df = df.rename(columns={"index": s_feat})

    df.columns = [s_feat, "Deprel", 'Proportion']
    df = df[df['Proportion'] != 0]
        
    # Calculate marginal probabilities
    g_marginals = df.groupby("Deprel")['Proportion'].sum().reset_index()
    g_marginals.columns = ["Deprel", 'Gram_Prob']

    s_marginals = df.groupby(s_feat)['Proportion'].sum().reset_index()
    s_marginals.columns = [s_feat, 'Sal_Prob']

    # Merge marginals with original DataFrame
    df = df.merge(g_marginals, on="Deprel")
    df = df.merge(s_marginals, on=s_feat)
 
    # Compute PMI
    df['PMI'] = np.log2(df['Proportion'] / (df['Gram_Prob'] * df['Sal_Prob']))
    df = df[df["Proportion"] >= 0.0001]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    
    df.dropna(inplace=True)
    df[s_feat] = df[s_feat].round(2)
    df["PMI"] = df["PMI"].round(2)
    df = df[df["Proportion"] >= 0.0001]
    print(df["PMI"].dtype)
    
 
    
    groups = df["Deprel"].unique()

    y_ticks = sorted(df[s_feat].unique())
    fig, axes = plt.subplots(1, len(groups), figsize=(4*len(groups), 4), sharey=True)
    y_ticks = sorted(df[s_feat].unique())
    y_map = {val: i for i, val in enumerate(y_ticks)}
    # Make sure axes is iterable
    if len(groups) == 1:
        axes = [axes]

    for ax, g in zip(axes, groups):
        subset = df[df["Deprel"] == g]
        ax.scatter(subset["PMI"], subset[s_feat].map(y_map), s=50)
        ax.set_title(f"Deprel = {g}")
        ax.set_xlabel("PMI")
        ax.set_xticks([-1,-0.5,0,0.5,1])  # show rounded PMI values
        ax.set_xticklabels([-1,-0.5,0,0.5,1], rotation=45)
        ax.axvline(x=0, color="grey", linestyle="--", linewidth=1)
        #ax.set_yticks(y_ticks)
        ax.set_yticks(range(len(y_ticks)))
        ax.set_yticklabels([str(v) for v in y_ticks])
        
    axes[0].set_ylabel(s_feat)

    plt.tight_layout()
    plt.show()

    


def plot_rank(s_feat,nb_bins,file_path,categ = "all"):
 
    d_count = get_feat_gram_dict(s_feat,"Rank",nb_bins,file_path,categ)
    df = pd.DataFrame.from_dict(d_count, orient='index')
    #existing_cols = [c for c in ["obl:agent","obj","nsubj:t","nsubj:i","nsubj:pass"] if c in df.columns]

    #df = df[existing_cols]
    df = df.fillna(0)
    # Calculate proportions
    grand_total = df.values.sum()
    print(df)
    # Calculate proportions
    df = df / grand_total
    
    # Reset index to get it in long form
    df = df.reset_index().melt(id_vars='index', var_name="Rank", value_name='Proportion')
    df = df.rename(columns={"index": s_feat})

    df.columns = [s_feat, "Rank", 'Proportion']
    df = df[df['Proportion'] != 0]
        
    # Calculate marginal probabilities
    g_marginals = df.groupby("Rank")['Proportion'].sum().reset_index()
    g_marginals.columns = ["Rank", 'Gram_Prob']

    s_marginals = df.groupby(s_feat)['Proportion'].sum().reset_index()
    s_marginals.columns = [s_feat, 'Sal_Prob']

    # Merge marginals with original DataFrame
    df = df.merge(g_marginals, on="Rank")
    df = df.merge(s_marginals, on=s_feat)
 
    # Compute PMI
    df['PMI'] = np.log2(df['Proportion'] / (df['Gram_Prob'] * df['Sal_Prob']))

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with NaN (or inf, now replaced with NaN)
    df.dropna(inplace=True)
    df[s_feat] = df[s_feat].round(2)
    df["PMI"] = df["PMI"].round(2)
    df = df[df["Proportion"] >= 0.0001]
    print(df)
    
    df["Rank_bin"] = np.where(df["Rank"] >= 4, "≥4", df["Rank"].astype(str))

    rank_bins = ["0", "1", "2", "3", "≥4"]

    # Distinct y ticks
    y_ticks = sorted(df[s_feat].unique())
    y_map = {val: i for i, val in enumerate(y_ticks)}

    # Create subplots
    fig, axes = plt.subplots(1, len(rank_bins), figsize=(4*len(rank_bins), 4), sharey=True)

    for ax, rb in zip(axes, rank_bins):
        subset = df[df["Rank_bin"] == rb]
        
        # Scatter
        ax.scatter(subset["PMI"], subset[s_feat].map(y_map), s=50, color="steelblue")
        
        # Vertical grey line at 0
        ax.axvline(x=0, color="grey", linestyle="--", linewidth=1)
        ax.set_yticks(range(len(y_ticks)))
        ax.set_yticklabels([str(v) for v in y_ticks])
        ax.set_title(f"Rank {rb}")
        ax.set_xlabel("PMI")
        ax.set_xticks([-1,-0.5,0,0.5,1])  # show rounded PMI values
        ax.set_xticklabels([-1,-0.5,0,0.5,1], rotation=45)
        #ax.set_yticks(y_ticks)

    axes[0].set_ylabel(s_feat)

    plt.tight_layout()
    plt.savefig("PMI_plots/rank_depth")
    plt.show()




    


def plot_number_many(s_feat,nb_bins,file_path):
    l_df = []
    for elem in ["all","animal","person","vehicle","food","outdoor"]:
        d_count = get_feat_gram_dict(s_feat,"number",nb_bins,file_path,categ = elem)
        df = pd.DataFrame.from_dict(d_count, orient='index')
        #existing_cols = [c for c in ["obl:agent","obj","nsubj:t","nsubj:i","nsubj:pass"] if c in df.columns]

        #df = df[existing_cols]
        df = df.fillna(0)
        # Calculate proportions
        grand_total = df.values.sum()
        print(df)
        # Calculate proportions
        df = df / grand_total
        
        # Reset index to get it in long form
        df = df.reset_index().melt(id_vars='index', var_name="Number", value_name='Proportion')
        df = df.rename(columns={"index": s_feat})

        df.columns = [s_feat, "number", 'Proportion']
        df = df[df['Proportion'] != 0]
            
        # Calculate marginal probabilities
        g_marginals = df.groupby("number")['Proportion'].sum().reset_index()
        g_marginals.columns = ["number", 'Gram_Prob']

        s_marginals = df.groupby(s_feat)['Proportion'].sum().reset_index()
        s_marginals.columns = [s_feat, 'Sal_Prob']

        # Merge marginals with original DataFrame
        df = df.merge(g_marginals, on="number")
        df = df.merge(s_marginals, on=s_feat)
    
        # Compute PMI
        df['PMI'] = np.log2(df['Proportion'] / (df['Gram_Prob'] * df['Sal_Prob']))

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop rows with NaN (or inf, now replaced with NaN)
        df.dropna(inplace=True)
        df[s_feat] = df[s_feat].round(2)
        df["PMI"] = df["PMI"].round(2)
        df = df[df["Proportion"] >= 0.0002]
        print(df)
        l_df.append(df)
    
    n_rows = len(l_df)
    fig, axes = plt.subplots(n_rows, 2, figsize=(6, 2 * n_rows), sharey=True)

    # Make axes 2D array even if n_rows == 1
    if n_rows == 1:
        axes = axes.reshape(1, 2)

    for i, df in enumerate(l_df):
        # Distinct y ticks
        y_ticks = sorted(df[s_feat].unique())
        y_map = {val: j for j, val in enumerate(y_ticks)}

        for j, rb in enumerate(["Sing", "Plur"]):
            ax = axes[i, j]
            subset = df[df["number"] == rb]

            # Scatter
            ax.scatter(subset["PMI"], subset[s_feat].map(y_map), s=50, color="steelblue")

            # Vertical grey line at 0
            ax.axvline(x=0, color="grey", linestyle="--", linewidth=1)

            # Titles / labels
            ax.set_yticks(range(len(y_ticks)))
            ax.set_yticklabels([str(v) for v in y_ticks])
            ax.set_title(f"{rb}")
            ax.set_xlabel("PMI")

        # Label y-axis only on the first column
        axes[i, 0].set_ylabel(s_feat)

    plt.tight_layout()
    plt.savefig("PMI_plots/nb_distance")
    plt.show()





def f(text, deprel,nlp):
    doc = nlp(text)
    if deprel in ["nsubj:i","nsubj:t"]:
        deprel = "nsubj"
    if deprel == "obj":
        deprel = "dobj"
    if deprel == "nsubj:pass":
        deprel = "nsubjpass"
    if deprel == "obl:agent":
        deprel = "pobj"
    for token in doc:
        #print(token.dep_)
        if token.pos_ == "NOUN" and token.dep_ == deprel:
            if token.tag_ == "NN":   # singular noun
                return "Sing"
            elif token.tag_ == "NNS":  # plural noun
                return "Plur"
    
    return None  # no match


def word_number(file_path):
    nlp = spacy.load("en_core_web_sm")
    df = pd.read_csv(file_path)
    tqdm.pandas()
    df["number"] = df.progress_apply(lambda row: f(row["caption"], row["Deprel"],nlp), axis=1)
    df.to_csv("coco_deprel_rank_no_hoi_coco_output_number.csv")



file_path = "localized_narratives_deprel_rank_no_hoi_coco_output.csv"
#s_feat = "Distance to center"
s_feat = "Size"
#s_feat = "Relative depth"
#g_feat = "number"

nb_bins = 5

##### Attention, agent not in  "coco_deprel_rank_no_hoi_coco_output_number.csv"

#plot_voice(s_feat,nb_bins,file_path)
#plot_voice(s_feat,nb_bins,file_path,"person")
#plot_voice(s_feat,nb_bins,file_path,"animal")
#plot_voice(s_feat,nb_bins,file_path,"vehicle")
#plot_voice(s_feat,nb_bins,file_path,"sports")
#word_number(file_path)

#nlp = spacy.load("en_core_web_sm")
#f("The mouse is eaten by my cat", "obj",nlp)