import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt





def reshape_df_for_heatmap_position(df, metric, log):
    min_val = df[metric].min()
    max_val = df[metric].max()
    if metric == "Depth":
        max_val = 1000
    if metric == "Relative depth":
        max_val = 3.5

    if log:
        bins = np.logspace(np.log10(min_val), np.log10(max_val), 10)
    else:
        bins = np.linspace(min_val, max_val, 10)
    df[metric + '_bin'] = pd.cut(df[metric], bins=bins)
    grouped = (
        df.groupby(['Super-category', metric + '_bin'])['Rank']
        .mean()
        .reset_index()
    )
    return grouped

def plot_comparison_heatmaps_position(df1, df2, metric, log,desired_order):
    #plt.rcParams['font.family'] = 'Times New Roman'
    sns.set_theme(style="whitegrid", palette="muted")

    df1_hm = reshape_df_for_heatmap_position(df1.copy(), metric, log)
    df2_hm = reshape_df_for_heatmap_position(df2.copy(), metric, log)

    pivot1 = df1_hm.pivot(index=metric + '_bin', columns="Super-category", values="Rank").iloc[::-1]
    pivot2 = df2_hm.pivot(index=metric + '_bin', columns="Super-category", values="Rank").iloc[::-1]

    # Reorder columns
    pivot1 = pivot1[[col for col in desired_order if col in pivot1.columns]]
    pivot2 = pivot2[[col for col in desired_order if col in pivot2.columns]]

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    sns.heatmap(pivot1, annot=True, fmt=".1f", linewidths=0.5, cmap="Blues",
                ax=axes[0])
    axes[0].set_title("COCO captions", fontsize=16)
    axes[0].set_xlabel("")
    axes[0].set_ylabel(metric + " in picture", fontsize=14)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), fontsize=12, rotation=45, ha='right')
    axes[0].set_yticklabels(axes[0].get_yticklabels(), fontsize=12)

    sns.heatmap(pivot2, annot=True, fmt=".1f", linewidths=0.5, cmap="Blues",
                ax=axes[1])
    axes[1].set_title("Localized narratives", fontsize=16)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")  # Already on left plot
    axes[1].set_xticklabels(axes[1].get_xticklabels(), fontsize=12, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f'plots_final/position_heatmap_comparison_{metric}.pdf', format='pdf')
    plt.show()
    #plt.close()






def reshape_df_for_heatmap_mentioned(df, metric, log):
    min_val = df[metric].min()
    max_val = df[metric].max()
    if metric == "Depth":
        max_val = 1000
    if metric == "Relative depth":
        max_val = 3.5
    if metric == "Size":
        max_val = 0.2

    if log:
        bins = np.logspace(np.log10(min_val), np.log10(max_val), 10)
    else: 
        bins = np.linspace(min_val, max_val, 10)
    df[metric + '_bin'] = pd.cut(df[metric], bins=bins)
    grouped = (
        df.groupby(['Super-category', metric + '_bin'])['Mentioned(%)']
        .mean()
        .reset_index()
    )
    return grouped

def plot_comparison_heatmaps_mentioned(df1, df2, metric, log, desired_order):
    #plt.rcParams['font.family'] = 'Times New Roman'
    sns.set_theme(style="whitegrid", palette="muted")

    df1_hm = reshape_df_for_heatmap_mentioned(df1.copy(), metric, log)
    df2_hm = reshape_df_for_heatmap_mentioned(df2.copy(), metric, log)

    pivot1 = df1_hm.pivot(index=metric + '_bin', columns="Super-category", values="Mentioned(%)").iloc[::-1]
    pivot2 = df2_hm.pivot(index=metric + '_bin', columns="Super-category", values="Mentioned(%)").iloc[::-1]



    # Reorder columns
    pivot1 = pivot1[[col for col in desired_order if col in pivot1.columns]]
    pivot2 = pivot2[[col for col in desired_order if col in pivot2.columns]]

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    sns.heatmap(pivot1, annot=True, fmt=".1f", linewidths=0.5, cmap="coolwarm",
                ax=axes[0])
    axes[0].set_title("COCO captions", fontsize=16)
    axes[0].set_xlabel("")
    axes[0].set_ylabel(metric + " in picture", fontsize=14)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), fontsize=12, rotation=45, ha='right')
    axes[0].set_yticklabels(axes[0].get_yticklabels(), fontsize=12)

    sns.heatmap(pivot2, annot=True, fmt=".1f", linewidths=0.5, cmap="coolwarm",
                ax=axes[1])
    axes[1].set_title("Localized narratives", fontsize=16)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")  # Already on left plot
    axes[1].set_xticklabels(axes[1].get_xticklabels(), fontsize=12, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f'plots_final/mentioned_heatmap_comparison_{metric}.pdf', format='pdf')
    plt.show()




def plot_HOI(df1,df2):

    # Convert HOI values to labels
    df1['HOI_label'] = df1['HOI'].map({1: 'yes', 0: 'no'})
    df2['HOI_label'] = df2['HOI'].map({1: 'yes', 0: 'no'})

    # Desired y-axis order
    desired_order = [
        'person', 'animal', 'vehicle', 'food', 'indoor', 'kitchen',
        'furniture', 'appliance', 'outdoor', 'sports', 'electronic', 'accessory'
    ]

    # Define hue order and palette
    hue_order = ['yes', 'no']
    palette = sns.color_palette("coolwarm", len(hue_order))
    darker_palette = [(r * 0.6, g * 0.6, b * 0.6) for r, g, b in palette]
    offset_map = {'yes': -0.2, 'no': 0.2}
    color_map = dict(zip(hue_order, darker_palette))

    # Create subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    datasets = [(df1, "COCO"), (df2, "Localized Narratives")]

    for ax, (df, title) in zip(axes, datasets):
        sns.violinplot(
            data=df,
            x="mentionned(%)",
            y="s_categ",
            hue="HOI_label",
            hue_order=hue_order,
            order=desired_order,
            palette=palette,
            scale="width",
            inner=None,
            cut=0,
            ax=ax
        )

        # Plot means as diamonds
        group_means = df.groupby(['s_categ', 'HOI_label'])['mentionned(%)'].mean().reset_index()
        y_pos_map = {cat: i for i, cat in enumerate(desired_order)}

        for _, row in group_means.iterrows():
            y_val = y_pos_map[row['s_categ']] + offset_map[row['HOI_label']]
            color = color_map[row['HOI_label']]
            ax.scatter(row['mentionned(%)'], y_val, marker='D', color=color, s=50, zorder=5, label='_nolegend_')

        ax.set_title(title)
        ax.xaxis.grid(True)
        ax.set(xlabel="mentionned(%)", ylabel="")
        sns.despine(trim=True, left=True)

    # Fix legend once
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[:2], labels[:2], title="HOI")

    # Save or show
    plt.tight_layout()
    plt.savefig("HOI_side_by_side.pdf", format="pdf")
    plt.show()


def deprel(df1,feat):
    
    palette = {
        'obl:agent': '#1f77b4',
        'nsubj:t': '#aec7e8',
        'nsubj:i': '#7f7f7f',
        'nsubj:pass': '#ff9896',
        'obj': '#d62728'
    }

    # Step 1: Filter relevant deprels
    target_deprels = ['nsubj:i', 'nsubj:t', 'nsubj:pass', 'obl:agent', 'obj']

    df1_filtered = df1[df1['Deprel'].isin(target_deprels)].copy()

    # Apply log-binning if the feature is 'Size'
    if feat == "Size":
        min_size = df1_filtered[feat].min()
        max_size = df1_filtered[feat].max()
        log_bins = np.logspace(np.log10(min_size), np.log10(max_size), num=11)  # 10 bins

        # Bin into log-spaced intervals
        df1_filtered['feat_bin'] = pd.cut(df1_filtered[feat], bins=log_bins, include_lowest=True)
    else:
        # For non-size features, just cut linearly into 10 bins
        df1_filtered['feat_bin'] = pd.qcut(df1_filtered[feat], q=10, duplicates='drop')
    group_counts = df1_filtered.groupby(['Super-category', 'feat_bin', 'Deprel']).size().reset_index(name='count')
    total_counts = df1_filtered.groupby(['Super-category', 'feat_bin']).size().reset_index(name='total')
    df_prop = pd.merge(group_counts, total_counts, on=['Super-category', 'feat_bin'])
    df_prop['proportion'] = df_prop['count'] / df_prop['total']

    # Optional: make size_bin readable
    #df_prop['feat_bin_str'] = df_prop['feat_bin'].astype(str)
    df_prop['feat_bin_str'] = df_prop['feat_bin'].apply(lambda x: f"{x.left:.2f}")

    # Step 4: Set up 4x3 FacetGrid
    g = sns.FacetGrid(
        df_prop,
        col='Super-category',
        col_wrap=4,
        height=3,
        sharey=True
    )

    # Step 5: Map line plot onto grid
    g.map_dataframe(
        sns.lineplot,
        x='feat_bin_str',
        y='proportion',
        hue='Deprel',
        marker='o',
        palette=palette
    )

    g.set_titles("{col_name}")
    g.set_axis_labels("Log-Spaced Feature Bin", "Proportion")

    for ax in g.axes.flatten():
        ax.tick_params(axis='x', rotation=45)

    # Step 6: Final touches
    g.set_axis_labels(feat+" Bin", "Proportion")
    g.set_titles(col_template="{col_name}")
    g.add_legend(title='DepRel')
    for ax in g.axes.flatten():
        ax.tick_params(axis='x', rotation=45)


    g._legend.set_bbox_to_anchor((1, 1))  # position top-right relative to figure
    g._legend.set_loc('upper right')


    plt.tight_layout()
    plt.show()
    plt.savefig("deprel"+feat+".pdf",format="pdf")

def plot_histo_HOI_feat(df,feat):
    df_grouped = df.groupby(['s_categ', 'HOI'])[feat].mean().unstack(fill_value=0)
    df_grouped['avg_total'] = (df_grouped[0] + df_grouped[1]) / 2
    df_grouped = df_grouped.sort_values(by='avg_total')
    df_grouped = df_grouped.drop(columns='avg_total')
    categories = df_grouped.index
    y_pos = np.arange(len(categories))
    bar_height = 0.4
    plt.figure(figsize=(10, 8))
    plt.barh(y_pos - bar_height/2, df_grouped[0], height=bar_height, color='skyblue', label='HOI = 0')
    plt.barh(y_pos + bar_height/2, df_grouped[1], height=bar_height, color='orange', label='HOI = 1')
    plt.yticks(y_pos, categories)
    plt.xlabel("Average "+feat)
    plt.title(f"Avg {feat} per Category (HOI = 0 vs HOI = 1)")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()




def main(to_plot):
    desired_order = ['person','animal', 'vehicle', 'food','furniture','appliance','electronic','sports','indoor', 'kitchen'
                        ,'outdoor', 'accessory' ]
    metrics = ["Colour saliency","Colour local contrast", "Distance to center","Depth"]

    if to_plot == "position_heatmap":
        df_coco = pd.read_csv('final results/coco_deprel_rank_no_hoi_coco_output.csv')
        df_ln = pd.read_csv('final results/localized_narratives_deprel_rank_no_hoi_coco_output.csv')
        
        df_coco = df_coco[df_coco['Rank'] != -1]
        df_ln = df_ln[df_ln['Rank'] != -1]
        
        for metric in metrics:
            plot_comparison_heatmaps_position(df_coco, df_ln, metric, log=False,desired_order=desired_order)
        plot_comparison_heatmaps_position(df_coco, df_ln, "Size", log=True,desired_order=desired_order)
    if to_plot == "mentioned_heatmap":
        df_coco = pd.read_csv("final results/cocomentioned_no_hoi_coco_output.csv")
        df_ln = pd.read_csv("final results/localized_narrativesmentioned_no_hoi_coco_output.csv")

        #for metric in metrics:
        #    plot_comparison_heatmaps_mentioned(df_coco, df_ln, metric, log=False,desired_order=desired_order)

        plot_comparison_heatmaps_mentioned(df_coco, df_ln, "Size", log=False,desired_order=desired_order)
    if to_plot == "HOI":
        df_coco = pd.read_csv('final results/coco_HOI_small.csv')
        df_ln = pd.read_csv('final results/localized_narratives_HOI_small.csv')
        plot_HOI(df_coco,df_ln)
    if to_plot == "deprel":
        df_coco = pd.read_csv('final results/coco_deprel_rank_no_hoi_coco_output.csv')
        #df_coco = df_coco[(df_coco['Distance to center'] >= 0.2) & (df_coco['Distance to center'] <= 0.3)]
        l_feats = ["Distance to center","Size"]#,"Colour local contrast","Depth",
                   #"Relative depth", "Colour saliency"]
        #df_coco['distance_times_size'] = (1-df_coco['Distance to center']) * np.log(df_coco['Size'])
        for feat in l_feats:
            deprel(df_coco,feat)
    if to_plot == "histo_avg_HOI":
        df_coco = pd.read_csv('final results/coco_HOI_small.csv')
        print(df_coco.columns)
        l_feat = ["size","distance_to_center","depth"]
        for feat in l_feat:
            plot_histo_HOI_feat(df_coco,feat)
        


    if to_plot == "histo_avg":
        feat = 'Depth'
        df_coco = pd.read_csv('final results/coco_deprel_rank_no_hoi_coco_output.csv')
        print(df_coco.columns)
        df_avg = df_coco.groupby('Super-category')[feat].mean()
        df_avg= df_avg.sort_values()
        df_avg.plot(kind='barh', figsize=(8, 6), color='skyblue')
        plt.xlabel("Average "+ feat)
        plt.title(f"Avg {feat} per Category")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

to_plot = "deprel"
main(to_plot)