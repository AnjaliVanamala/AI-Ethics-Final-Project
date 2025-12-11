import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def set_plot_style():
    """
    Configure global matplotlib / seaborn style for print-quality figures.
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "figure.autolayout": False,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
        "font.size": 11,
    })

set_plot_style()


def clean_data(df):
    """
    Cleans and standardizes the values in the dataframe.
    """
    # Standardize Gender
    gender_map = {
        'Male': 'Male',
        'Female': 'Female',
        'Male, Female': 'Both',
        'Male and Female': 'Both',
        'Not determinable': 'Undeterminable',
        'Undeterminable': 'Undeterminable',
        'Not Inferable': 'Undeterminable',
        'Cannot Determine': 'Undeterminable',
        'Unspecified': 'Undeterminable'
    }
    df['Gender'] = df['Gender'].map(gender_map).fillna('Undeterminable')

    # Standardize other columns (handle NaNs)
    cols_to_fix = ['Socioeconomic Status', 'Geographic Location', 'Cultural Background']
    for col in cols_to_fix:
        df[col] = df[col].fillna('Undeterminable')

    if 'domain' in df.columns:
        df['domain'] = df['domain'].fillna('Undeterminable')
        
    return df


def sort_values_with_undeterminable_last(values):
    """
    Sorts values alphabetically but puts 'Undeterminable' at the end.
    """
    values = sorted(list(values))
    if 'Undeterminable' in values:
        values.remove('Undeterminable')
        values.append('Undeterminable')
    return values


def load_data():
    dfs = []
    # Map filenames to Disability names
    files = {
        "Answers/Disability 1_results.csv": "Autism",
        "Answers/Disability 2_results.csv": "Dyslexia",
        "Answers/Disability 3_results.csv": "ADHD",
        "Answers/Neutral Query_results.csv": "None"
    }
    
    for filepath, disability_name in files.items():
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                df['Disability'] = disability_name
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
    
    if not dfs:
        print("No data files found.")
        return None
        
    full_df = pd.concat(dfs, ignore_index=True)
    full_df = clean_data(full_df)
    return full_df


def plot_attribute_distribution_by_model_and_disability(df, attribute, output_dir):
    """
    Plots the distribution of an attribute for each model, faceted by Disability using stacked bars.
    """
    print(f"Plotting distribution for {attribute}...")
    
    disabilities = sorted(df['Disability'].unique())
    n_cols = len(disabilities)
    
    total_cols = n_cols + 1
    width_ratios = [6] * n_cols + [1]
    
    fig_width = 3 * n_cols + 1
    fig_height = 4
    fig, axes = plt.subplots(
        1,
        total_cols,
        figsize=(fig_width, fig_height),
        sharey=True,
        gridspec_kw={"width_ratios": width_ratios}
    )
    
    plot_axes = axes[:-1]
    legend_ax = axes[-1]
    if n_cols == 1:
        plot_axes = [plot_axes]
        
    unique_values = sort_values_with_undeterminable_last(df[attribute].dropna().unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_values)))
    color_map = dict(zip(unique_values, colors))

    for ax, disability in zip(plot_axes, disabilities):
        subset = df[df['Disability'] == disability]
        
        ct = pd.crosstab(subset['model'], subset[attribute], normalize='index')
        ct = ct.reindex(columns=unique_values, fill_value=0)
        
        ct.plot(
            kind='bar',
            stacked=True,
            ax=ax,
            color=[color_map[col] for col in ct.columns],
            width=0.6
        )
        
        ax.set_title(f'{disability}', fontsize=11)
        ax.set_xlabel('Model')
        if ax is plot_axes[0]:
            ax.set_ylabel('Percentage')
        else:
            ax.set_ylabel('')
            
        ax.tick_params(axis='x', rotation=0)

        if ax.get_legend():
            ax.get_legend().remove()

    legend_ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[v]) for v in unique_values]
    legend_ax.legend(
        handles,
        unique_values,
        title=attribute,
        loc="center left",
        frameon=False
    )
    
    fig.suptitle(f'Distribution of {attribute} by Model and Disability', fontsize=12)
    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.92])
    fig.savefig(f'{output_dir}/{attribute}_distribution_by_model.pdf')
    plt.close(fig)


def plot_attribute_distribution_overall(df, attribute, output_dir):
    """
    Plots the overall distribution of an attribute across all data.
    """
    plt.figure(figsize=(4,3))
    
    counts = df[attribute].value_counts(normalize=True).reset_index()
    counts.columns = [attribute, 'percentage']
    
    sns.barplot(data=counts, x=attribute, y='percentage', palette='magma', width=0.6)
    plt.title(f'Overall Distribution of {attribute}', fontsize=12)
    plt.ylabel('Percentage')
    plt.xlabel(attribute)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{attribute}_overall_distribution.pdf', bbox_inches='tight')
    plt.close()


def analyze_correlations(df, output_dir):
    """
    Analyzes and plots the relationship between Disability and Attributes.
    """
    print("Analyzing correlations...")
    attributes = ['Age', 'Gender', 'Socioeconomic Status', 'Geographic Location',
                  'Educational Background', 'Cultural Background']
    
    for attribute in attributes:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        contingency_table = pd.crosstab(df['Disability'], df[attribute], normalize='index')
        
        sns.heatmap(
            contingency_table,
            annot=True,
            cmap="YlGnBu",
            fmt=".2f",
            annot_kws={"size": 12}
        )
        ax.set_title(f'Association: Disability vs {attribute}', fontsize=12)
        ax.set_ylabel('Disability')
        ax.set_xlabel(attribute)
        ax.tick_params(axis='x', rotation=0, labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        fig.tight_layout()
        fig.savefig(f'{output_dir}/Heatmap_Disability_vs_{attribute}.pdf', bbox_inches='tight')
        plt.close(fig)


def plot_model_vs_attribute_heatmap(df, attribute, output_dir):
    """
    Plots a heatmap showing the relationship between Models and an Attribute.
    """
    plt.figure(figsize=(5,4.5))
    contingency_table = pd.crosstab(df['model'], df[attribute], normalize='index')
    
    sns.heatmap(
        contingency_table,
        annot=True,
        cmap="YlGnBu",
        fmt=".2f",
        annot_kws={"size": 12}
    )
    plt.title(f'Model vs {attribute}', fontsize=12)
    plt.ylabel('Model')
    plt.xlabel(attribute)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Heatmap_Model_vs_{attribute}.pdf', bbox_inches='tight')
    plt.close()


def plot_model_vs_attribute_stacked(df, attribute, output_dir):
    """
    Plots the distribution of an attribute for each model (aggregated across all disabilities).
    """
    unique_values = sort_values_with_undeterminable_last(df[attribute].dropna().unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_values)))
    color_map = dict(zip(unique_values, colors))
    
    ct = pd.crosstab(df['model'], df[attribute], normalize='index')
    ct = ct.reindex(columns=unique_values, fill_value=0)
    
    fig, (ax_plot, ax_legend) = plt.subplots(
        1, 2,
        figsize=(5, 3),
        gridspec_kw={'width_ratios': [4, 1]}
    )
    
    ct.plot(
        kind='bar',
        stacked=True,
        color=[color_map[col] for col in ct.columns],
        width=0.6,
        ax=ax_plot
    )
    
    ax_plot.set_title(f'Model Bias on {attribute}', fontsize=12)
    ax_plot.set_xlabel('Model')
    ax_plot.set_ylabel('Percentage')
    ax_plot.tick_params(axis='x', rotation=0)

    ax_legend.axis('off')
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[v]) for v in unique_values]
    handles, labels = ax_plot.get_legend_handles_labels()
    if ax_plot.get_legend() is not None:
        ax_plot.get_legend().remove()
    ax_legend.legend(
        handles,
        unique_values,
        title=attribute,
        loc='center left',
        frameon=False
    )
    
    fig.tight_layout()
    fig.savefig(f'{output_dir}/{attribute}_model_bias_stacked.pdf')
    plt.close(fig)


def plot_domain_vs_attribute_heatmap(df, attribute, output_dir):
    """Plots normalized heatmap of domain vs attribute."""
    if 'domain' not in df.columns:
        print("Skipping domain heatmap; 'domain' column missing.")
        return

    plt.figure(figsize=(5, 4.5))
    contingency_table = pd.crosstab(df['domain'], df[attribute], normalize='index')

    sns.heatmap(
        contingency_table,
        annot=True,
        cmap="YlGnBu",
        fmt=".2f",
        annot_kws={"size": 12}
    )
    plt.title(f'Domain vs {attribute}', fontsize=12)
    plt.ylabel('Domain')
    plt.xlabel(attribute)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Heatmap_Domain_vs_{attribute}.pdf', bbox_inches='tight')
    plt.close()


def plot_domain_vs_attribute_stacked(df, attribute, output_dir):
    """Plots stacked share of attribute values within each domain."""
    if 'domain' not in df.columns:
        print("Skipping domain stacked chart; 'domain' column missing.")
        return

    unique_values = sort_values_with_undeterminable_last(df[attribute].dropna().unique())
    colors = plt.cm.cividis(np.linspace(0, 1, len(unique_values)))
    color_map = dict(zip(unique_values, colors))

    ct = pd.crosstab(df['domain'], df[attribute], normalize='index')
    ct = ct.reindex(columns=unique_values, fill_value=0)

    fig_width = max(5, 0.6 * len(ct.index) + 2)
    fig, (ax_plot, ax_legend) = plt.subplots(
        1, 2,
        figsize=(fig_width, 3.2),
        gridspec_kw={'width_ratios': [4, 1]}
    )

    ct.plot(
        kind='bar',
        stacked=True,
        color=[color_map[col] for col in ct.columns],
        width=0.7,
        ax=ax_plot
    )
    if ax_plot.get_legend() is not None:
        ax_plot.get_legend().remove()

    ax_plot.set_title(f'Domain Bias on {attribute}', fontsize=12)
    ax_plot.set_xlabel('Domain')
    ax_plot.set_ylabel('Percentage')
    ax_plot.tick_params(axis='x', rotation=45 if len(ct.index) > 3 else 0)
    if len(ct.index) > 3:
        plt.setp(ax_plot.get_xticklabels(), ha='right')

    ax_legend.axis('off')
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[v]) for v in unique_values]
    ax_legend.legend(
        handles,
        unique_values,
        title=attribute,
        loc='center left',
        frameon=False
    )

    fig.tight_layout()
    fig.savefig(f'{output_dir}/{attribute}_domain_bias_stacked.pdf')
    plt.close(fig)


def plot_domain_distribution_overall(df, output_dir):
    """Plots how frequently each domain appears across the full dataset."""
    if 'domain' not in df.columns:
        print("Skipping domain plots; 'domain' column missing.")
        return

    plt.figure(figsize=(5, 3.2))
    counts = df['domain'].value_counts(normalize=True).reset_index()
    counts.columns = ['domain', 'percentage']

    sns.barplot(data=counts, x='domain', y='percentage', palette='crest', width=0.6)
    plt.title('Overall Distribution of Domains', fontsize=12)
    plt.ylabel('Percentage')
    plt.xlabel('Domain')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Domain_overall_distribution.pdf', bbox_inches='tight')
    plt.close()


def plot_domain_distribution_by_category(df, category, output_dir):
    """Plots domain share for each value inside the selected category column."""
    if 'domain' not in df.columns or category not in df.columns:
        print(f"Skipping domain plot by {category}; required columns missing.")
        return

    unique_domains = sort_values_with_undeterminable_last(df['domain'].dropna().unique())
    colors = plt.cm.plasma(np.linspace(0, 1, len(unique_domains)))
    color_map = dict(zip(unique_domains, colors))

    ct = pd.crosstab(df[category], df['domain'], normalize='index')
    ct = ct.reindex(columns=unique_domains, fill_value=0).sort_index()

    fig_width = max(5, 0.6 * len(ct.index) + 2)
    fig, (ax_plot, ax_legend) = plt.subplots(
        1,
        2,
        figsize=(fig_width, 3.5),
        gridspec_kw={'width_ratios': [4, 1]}
    )

    ct.plot(
        kind='bar',
        stacked=True,
        color=[color_map[col] for col in ct.columns],
        width=0.7,
        ax=ax_plot
    )
    if ax_plot.get_legend() is not None:
        ax_plot.get_legend().remove()

    category_label = category.title()
    ax_plot.set_title(f'Domain Distribution by {category_label}', fontsize=12)
    ax_plot.set_xlabel(category_label)
    ax_plot.set_ylabel('Percentage')
    rotation = 45 if len(ct.index) > 3 else 0
    ax_plot.tick_params(axis='x', rotation=rotation)
    if rotation:
        plt.setp(ax_plot.get_xticklabels(), ha='right')

    ax_legend.axis('off')
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[v]) for v in unique_domains]
    ax_legend.legend(
        handles,
        unique_domains,
        title='Domain',
        loc='center left',
        frameon=False
    )

    safe_category = category_label.replace(' ', '_')
    fig.tight_layout()
    fig.savefig(f'{output_dir}/Domain_distribution_by_{safe_category}.pdf')
    plt.close(fig)


def generate_summary_tables(df, output_dir):
    """
    Generates CSV summary tables.
    """
    print("Generating summary tables...")
    attributes = ['Age', 'Gender', 'Socioeconomic Status', 'Geographic Location',
                  'Educational Background', 'Cultural Background']
    
    # 1. Summary by Model and Disability
    for attribute in attributes:
        summary = df.groupby(['model', 'Disability', attribute]).size().unstack(fill_value=0)
        summary_pct = summary.div(summary.sum(axis=1), axis=0)
        summary_pct.to_csv(f'{output_dir}/Summary_{attribute}_by_Model_Disability.csv')

    # 2. Summary by Disability only
    for attribute in attributes:
        summary = df.groupby(['Disability', attribute]).size().unstack(fill_value=0)
        summary_pct = summary.div(summary.sum(axis=1), axis=0)
        summary_pct.to_csv(f'{output_dir}/Summary_{attribute}_by_Disability.csv')


def main():
    output_dir = "Analysis_Results"
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_data()
    if df is None:
        return

    # Save combined raw data
    df.to_csv(f'{output_dir}/combined_data.csv', index=False)

    attributes = ['Age', 'Gender', 'Socioeconomic Status', 'Geographic Location',
                  'Educational Background', 'Cultural Background']
    
    # 1. Analyze prediction bias (Visualizations)
    for attr in attributes:
        plot_attribute_distribution_by_model_and_disability(df, attr, output_dir)
        plot_attribute_distribution_overall(df, attr, output_dir)
        plot_model_vs_attribute_heatmap(df, attr, output_dir)
        plot_model_vs_attribute_stacked(df, attr, output_dir)
        plot_domain_vs_attribute_heatmap(df, attr, output_dir)
        plot_domain_vs_attribute_stacked(df, attr, output_dir)

    # 1b. Domain-focused visualizations
    plot_domain_distribution_overall(df, output_dir)
    plot_domain_distribution_by_category(df, 'model', output_dir)
    plot_domain_distribution_by_category(df, 'Disability', output_dir)
        
    # 2. Correlation Analysis (Heatmaps)
    analyze_correlations(df, output_dir)
    
    # 3. Generate CSV Tables
    generate_summary_tables(df, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()