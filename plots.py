import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from object_detection import prop_of_detected_w_are_mentionned
from matplotlib.lines import Line2D
import numpy as np



def plot_size_pos(csv_file):

    df = pd.read_csv(csv_file) 
    unique_labels = df['label'].unique()
    colors = plt.cm.get_cmap('jet', len(unique_labels)) 
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(unique_labels):
        subset = df[df['label'] == label]  
        plt.scatter(subset['position'], subset['size'], label=label, color=colors(i))  
    plt.legend(title="Label")
    plt.xlabel('Position in sentence')
    plt.ylabel('Size in picture')
    plt.show()

def plot_metric_pos_bins(csv_file,rel,metric):

    df = pd.read_csv(csv_file) 
    df = reshape_df_for_dot_plot_pos(df,rel,metric)
    
    # Manually define the desired order of labels
    desired_order = ['person','animal', 'vehicle', 'indoor', 'food', 'kitchen','furniture','appliance','outdoor', 'sports','electronic', 'accessory' ]
    
    colors = plt.cm.get_cmap('jet_r', len(desired_order)) 
    
    plt.figure(figsize=(8, 6))

    for i, label in enumerate(desired_order):
        if label in df['label'].unique():  # Check if the label exists in the data
            subset = df[df['label'] == label]  
            plt.scatter(subset['pos_bin_mid'], subset[metric+'_bin_mid'], label=label, color=colors(i))
            if rel:  
                avg_pos = subset['pos_bin_mid'].mean()
                plt.axvline(x=avg_pos, color=colors(i), linestyle='--', linewidth=2,label='_nolegend_')
    
    plt.legend(title="Label", labels=desired_order)
    plt.xlabel('Position in sentence')
    plt.ylabel(metric+' in picture')
    plt.show()

def plot_metric_rank_bins(csv_file,rel,metric):

    df = pd.read_csv(csv_file) 
    df = reshape_df_for_dot_plot_rank(df,rel,metric)
    
    # Manually define the desired order of labels
    desired_order = ['person','animal', 'vehicle', 'indoor', 'food', 'kitchen','furniture','appliance','outdoor', 'sports','electronic', 'accessory' ]
    
    colors = plt.cm.get_cmap('jet_r', len(desired_order)) 
    
    plt.figure(figsize=(8, 6))

    for i, label in enumerate(desired_order):
        if label in df['label'].unique():  # Check if the label exists in the data
            subset = df[df['label'] == label]  
            plt.scatter(subset['rank_bin_mid'], subset[metric+'_bin_mid'], label=label, color=colors(i),alpha=0.8) 
            if rel: 
                avg_rank = subset['rank_bin_mid'].mean()
                plt.axvline(x=avg_rank, color=colors(i), linestyle='--', linewidth=2,label='_nolegend_')
    
    plt.legend(title="Label", labels=desired_order)
    plt.xlabel('Rank of noun in sentence')
    plt.ylabel(metric+' in picture')
    plt.show()


def plot_hoi(csv_file):
    df = pd.read_csv(csv_file) 
    df = df[~df['label'].isin(["person","electronic","applience"])]

    desired_order = ['animal', 'vehicle', "food",'outdoor', "indoor",'kitchen','furniture', 'sports','accessory' ]
    df["label"] = pd.Categorical(df['label'],
                                   categories=desired_order,
                                   ordered=True)
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=df,
        x='label',           
        y="mentionned(%)",        
        hue='HOI',        
        #errorbar='sd',            
        palette='Set2',
    )

    plt.title("% mention per Label")
    plt.ylabel("% mention")
    plt.xlabel("Category")
    plt.show()  



def reshape_df_for_dot_plot_pos(df,rel,metric):
    nb_bins = 70
    min_size = df[metric].min()
    max_size = df[metric].max()
    if metric == "saliency_rel":
        max_size = 5
    if rel:
        pos_colum = 'position_rel'
    else:
        pos_colum = 'position'
    min_pos = df[pos_colum].min()
    max_pos = df[pos_colum].max()

    bins_size = np.linspace(min_size,max_size, nb_bins)  
    bins_pos = np.linspace(min_pos,max_pos, nb_bins) 
    df[metric+'_bin'] = pd.cut(df[metric], bins=bins_size)
    df['pos_bin'] = pd.cut(df[pos_colum], bins=bins_pos)

    grouped = (
        df.groupby(['label',metric+'_bin', 'pos_bin'], as_index=False, observed=True)
        .agg({metric: 'mean', pos_colum: 'mean'})
        .dropna()
    )

    grouped['pos_bin_mid'] = grouped['pos_bin'].apply(lambda x: float(x.mid)).astype(float)
    grouped[metric+'_bin_mid'] = grouped[metric+'_bin'].apply(lambda x: float(x.mid)).astype(float)

    noise_size = np.random.normal(loc=0, scale=1/(2*nb_bins), size=grouped[metric+'_bin_mid'].shape)
    noise_pos = np.random.normal(loc=0, scale=1/(2*nb_bins), size=grouped['pos_bin_mid'].shape)
    grouped[metric+'_bin_mid'] = grouped[metric+'_bin_mid'] + noise_size
    grouped['pos_bin_mid'] = grouped['pos_bin_mid'] + noise_pos

    return grouped



def reshape_df_for_dot_plot_rank(df,rel,metric):
    nb_bins_size = 40
    nb_bins_rank = 120

    if rel:
        rank_colum = 'rank_rel'
    else:
        rank_colum = 'rank'

    min_size = df[metric].min()
    max_size = df[metric].max()
    if metric == "saliency_rel":
        max_size = 5

    min_rank = df[rank_colum].min()
    max_rank = df[rank_colum].max()

    bins_size = np.linspace(min_size,max_size, nb_bins_size)  
    bins_rank = np.linspace(min_rank,max_rank, nb_bins_rank) 
    df[metric+'_bin'] = pd.cut(df[metric], bins=bins_size)
    df['rank_bin'] = pd.cut(df[rank_colum], bins=bins_rank)

    grouped = (
        df.groupby(['label',metric+'_bin', 'rank_bin'], as_index=False, observed=True)
        .agg({metric: 'mean', rank_colum: 'mean'})
        .dropna()
    )

    grouped['rank_bin_mid'] = grouped['rank_bin'].apply(lambda x: float(x.mid)).astype(float)
    grouped[metric+'_bin_mid'] = grouped[metric+'_bin'].apply(lambda x: float(x.mid)).astype(float)

    noise_size = np.random.normal(loc=0, scale=1/(2*nb_bins_size), size=grouped[metric+'_bin_mid'].shape)
    noise_pos = np.random.normal(loc=0, scale=1/(2*nb_bins_rank), size=grouped['rank_bin_mid'].shape)
    grouped[metric+'_bin_mid'] = grouped[metric+'_bin_mid'] + noise_size
    grouped['rank_bin_mid'] = grouped['rank_bin_mid'] + noise_pos

    return grouped


def plot_size_pos_line_regression(csv_file):
    df = pd.read_csv(csv_file) 
    unique_labels = df['label'].unique()
    colors = plt.cm.get_cmap('jet', len(unique_labels)) 
    plt.figure(figsize=(8, 6))

    for i, label in enumerate(unique_labels):
        subset = df[df['label'] == label]  
        x = subset['position']
        y = subset['size']
        
        # Scatter plot
        plt.scatter(x, y, label=label, color=colors(i))  
        
        # Fit and plot regression line
        if len(subset) >= 2:  # Need at least 2 points
            coef = np.polyfit(x, y, 1)  # Linear fit: degree 1
            poly1d_fn = np.poly1d(coef)
            x_sorted = np.sort(x)
            plt.plot(x_sorted, poly1d_fn(x_sorted), color=colors(i), linestyle='--')

    plt.legend(title="Label")
    plt.xlabel('Position in sentence')
    plt.ylabel('Size in picture')
    plt.title("Size vs Position with Regression Lines")
    plt.tight_layout()
    plt.show()


def plot_size_prop(csv_file):
    df = pd.read_csv(csv_file)

    mention_order = [f"{k}/5" for k in range(0, 6)] 

    sns.set_theme(style="whitegrid", palette="muted")

    plt.figure(figsize=(10, 6))
    ax = sns.swarmplot(
        data=df,
        x="mentionned(%)",
        y="size",
        hue="label",
        dodge=True,
        order=mention_order 
    )

    ax.set(xlabel="Mentioned in captions (%)", ylabel="Size in picture")
    plt.legend(title="Label")
    plt.tight_layout()
    plt.show()



def plot_size_pos_nb(csv_file):
    df = pd.read_csv(csv_file) 
    unique_labels = df['label'].unique()
    colors = plt.cm.get_cmap('jet', len(unique_labels)) 

    markers = {'one': 'o', 'several': '^','crowd':'*'}  # you can extend this

    plt.figure(figsize=(8, 6))

    for i, label in enumerate(unique_labels):
        color = colors(i)
        for number, marker in markers.items():
            subset = df[(df['label'] == label) & (df['number'] == number)]
            if not subset.empty:
                plt.scatter(
                    subset['position'], 
                    subset['size'], 
                    color=color, 
                    marker=marker,
                    alpha=0.7
                )

    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label=label,
               markerfacecolor=colors(i), markersize=10)
        for i, label in enumerate(unique_labels)
    ]

    gender_legend = [
        Line2D([0], [0], marker=marker, color='k', label=number, linestyle='None', markersize=8)
        for number, marker in markers.items()
    ]

    plt.legend(handles=legend_elements + gender_legend, title="Labels + Number", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('Position in sentence')
    plt.ylabel('Size in picture')
    plt.title('Size vs Position by Label and Number')
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    


def barplot(csv_file):
    df = pd.read_csv(csv_file)
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=df,
        x='label',           
        y='position',        
        hue='number',        
        errorbar='sd',            
        palette='Set2'
    )

    plt.title("Average Position per Label and number")
    plt.ylabel("Average Position")
    plt.xlabel("Label")
    plt.show()



def reshape_df_for_heatmap(df,metric="size",log=True):
    print(metric)
    min_val = df[metric].min()
    max_val = df[metric].max()
    if metric == "depth":
        max_val = 1000
    if metric == "depth_rel":
        max_val = 3.5

    if log:
        bins = np.logspace(np.log10(min_val), np.log10(max_val), 19)  
    else:
        bins = np.linspace(min_val, max_val, 19)
    print(max_val,bins)
    df[metric+'_bin'] = pd.cut(df[metric], bins=bins)
    grouped = (
        df.groupby(['label', metric+'_bin'])['mentionned(%)']
        .mean()
        .reset_index()
        #.rename(columns={'mentionned(%)': 'mentionned(%)'})
        #.dropna()
    )
    return grouped


def plot_heatmap_prop_mentionned(csv_file,metric,log=False):
    df = pd.read_csv(csv_file) 
    df = reshape_df_for_heatmap(df,metric)
    sns.set_theme(style="whitegrid", palette="muted")

    df = (
        df
        .pivot(index=metric+'_bin', columns="label", values="mentionned(%)").iloc[::-1]
    )
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(df, annot=True, linewidths=.5,cmap ="coolwarm" )

    ax.set(xlabel="Mentioned in captions (%)", ylabel=metric+" in picture")
    plt.legend(title="Label")
    plt.tight_layout()
    plt.show()
