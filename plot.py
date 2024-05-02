import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the plot directory exists
plot_dir = './plot'
os.makedirs(plot_dir, exist_ok=True)

results_df = pd.read_csv('results.csv')

results_df['metrics'] = results_df['metrics'].apply(eval)

results_df['RMSE'] = results_df['metrics'].apply(lambda x: x['RMSE'])
results_df['MAE'] = results_df['metrics'].apply(lambda x: x['MAE'])

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.5})
plt.rcParams["figure.figsize"] = (12, 8)

plt.figure(figsize=(12, 8))
sns.lineplot(x='sketch_size', y='RMSE', hue='algorithm', data=results_df[results_df['dataset'] == '20newsgroups'],
             marker='o')
plt.xlabel('Sketch Size')
plt.ylabel('RMSE')
plt.title('Algorithm Performance on 20newsgroups Dataset (RMSE)')
plt.legend(title='Algorithm', loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '20newsgroups_performance_rmse.png'), dpi=300, bbox_inches='tight')

plt.figure(figsize=(12, 8))
sns.lineplot(x='sketch_size', y='MAE', hue='algorithm', data=results_df[results_df['dataset'] == '20newsgroups'],
             marker='o')
plt.xlabel('Sketch Size')
plt.ylabel('MAE')
plt.title('Algorithm Performance on 20newsgroups Dataset (MAE)')
plt.legend(title='Algorithm', loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '20newsgroups_performance_mae.png'), dpi=300, bbox_inches='tight')

plt.figure(figsize=(18, 6))
g = sns.relplot(x='sketch_size', y='RMSE', hue='algorithm', col='sparsity',
                data=results_df[results_df['dataset'] == 'generated'],
                kind='line', marker='o',
                facet_kws={'sharex': False, 'sharey': True, 'legend_out': True},
                col_wrap=3, height=5, aspect=1.2)

g.set(yscale='log', xlabel='Sketch Size', ylabel='RMSE')
g.fig.suptitle('Algorithm Performance on Generated Dataset (RMSE)', fontsize=18, y=1.05)
g.fig.subplots_adjust(top=0.8, wspace=0.2, hspace=0.2)
legend = g.axes[-1].legend(bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0.)
legend.get_frame().set_alpha(0.8)
for i, ax in enumerate(g.axes):
    ax.set_title(f'Sparsity: {ax.get_title().split("=")[1].strip()}', fontsize=14)
    if i % 3 != 0:
        ax.set_ylabel('')
    ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'generated_dataset_rmse.png'), dpi=300, bbox_inches='tight')

plt.figure(figsize=(18, 6))
g = sns.relplot(x='sketch_size', y='MAE', hue='algorithm', col='sparsity',
                data=results_df[results_df['dataset'] == 'generated'],
                kind='line', marker='o',
                facet_kws={'sharex': False, 'sharey': True, 'legend_out': True},
                col_wrap=3, height=5, aspect=1.2)

g.set(yscale='log', xlabel='Sketch Size', ylabel='MAE')
g.fig.suptitle('Algorithm Performance on Generated Dataset (MAE)', fontsize=18, y=1.05)
g.fig.subplots_adjust(top=0.8, wspace=0.2, hspace=0.2)
legend = g.axes[-1].legend(bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0.)
legend.get_frame().set_alpha(0.8)
for i, ax in enumerate(g.axes):
    ax.set_title(f'Sparsity: {ax.get_title().split("=")[1].strip()}', fontsize=14)
    if i % 3 != 0:
        ax.set_ylabel('')
    ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'generated_dataset_mae.png'), dpi=300, bbox_inches='tight')

plt.figure(figsize=(12, 8))
sns.lineplot(x='sketch_size', y='avg_execution_time', hue='algorithm',
             data=results_df[results_df['dataset'] == '20newsgroups'], marker='o')
plt.title('Algorithm Execution Time Comparison on 20newsgroups Dataset')
plt.xlabel('Sketch Size')
plt.ylabel('Execution Time (s)')
plt.yscale('log')
plt.legend(title='Algorithm', loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'algorithm_execution_time_20newsgroups.png'), dpi=300, bbox_inches='tight')

median_sketch_size = results_df['sketch_size'].median()

plt.figure(figsize=(12, 8))
sns.lineplot(x='sparsity', y='avg_execution_time', hue='algorithm',
             data=results_df[(results_df['dataset'] == 'generated') & (results_df['sketch_size'] == median_sketch_size)],
             marker='o')
plt.title(f'Algorithm Execution Time Comparison on Generated Dataset (Sketch Size: {median_sketch_size})')
plt.xlabel('Sparsity')
plt.ylabel('Execution Time (s)')
plt.yscale('log')
plt.legend(title=f'Algorithm (Sketch Size: {median_sketch_size})', loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f'algorithm_execution_time_generated_sketchsize_{median_sketch_size}.png'), dpi=300, bbox_inches='tight')