import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取实验结果文件
results_df = pd.read_csv('results.csv')

# 将metrics列从字符串转换为字典
results_df['metrics'] = results_df['metrics'].apply(eval)

# 提取RMSE和MAE值到单独的列
results_df['RMSE'] = results_df['metrics'].apply(lambda x: x['RMSE'])
results_df['MAE'] = results_df['metrics'].apply(lambda x: x['MAE'])

# 设置图表样式
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.5})
plt.rcParams["figure.figsize"] = (12, 8)

# 算法在20newsgroups数据集上的性能比较图
# 创建一个包含两个子图的图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
# 绘制RMSE子图
sns.lineplot(x='sketch_size', y='RMSE', hue='algorithm', data=results_df[results_df['dataset'] == '20newsgroups'],
             marker='o', ax=ax1)
ax1.set_xlabel('Sketch Size')
ax1.set_ylabel('RMSE')
ax1.set_title('RMSE Performance on 20newsgroups Dataset')
ax1.legend(title='Algorithm', loc='upper right')
# 绘制MAE子图
sns.lineplot(x='sketch_size', y='MAE', hue='algorithm', data=results_df[results_df['dataset'] == '20newsgroups'],
             marker='o', ax=ax2)
ax2.set_xlabel('Sketch Size')
ax2.set_ylabel('MAE')
ax2.set_title('MAE Performance on 20newsgroups Dataset')
ax2.legend(title='Algorithm', loc='upper right')
plt.tight_layout()
plt.savefig('20newsgroups_performance.png', dpi=300)

# 算法在自己生成的数据集上的性能比较图 (RMSE)
plt.figure(figsize=(18, 6))
g = sns.relplot(x='sketch_size', y='RMSE', hue='algorithm', col='sparsity',
                data=results_df[results_df['dataset'] == 'generated'],
                kind='line', marker='o',
                facet_kws={'sharex': False, 'sharey': True, 'legend_out': True},
                col_wrap=3, height=5, aspect=1.2)

g.set(yscale='log', xlabel='Sketch Size', ylabel='RMSE')  # 设置 y 轴为对数刻度, 并添加轴标签
g.fig.suptitle('Algorithm Performance on Generated Dataset (RMSE)', fontsize=18, y=1.05)
g.fig.subplots_adjust(top=0.8, wspace=0.2, hspace=0.2)

# 调整图例位置
legend = g.axes[-1].legend(bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0.)
legend.get_frame().set_alpha(0.8)  # 设置图例背景的透明度

for i, ax in enumerate(g.axes):
    ax.set_title(f'Sparsity: {ax.get_title().split("=")[1].strip()}', fontsize=14)  # 调整子图标题的格式和字体大小

    if i % 3 != 0:  # 除了第一列的子图,其他子图隐藏y轴标签
        ax.set_ylabel('')

    ax.grid(True, linestyle='--', alpha=0.7)  # 添加网格线

plt.tight_layout()
plt.savefig('generated_dataset_rmse.png', dpi=300, bbox_inches='tight')

# 算法在自己生成的数据集上的性能比较图 (MAE)
plt.figure(figsize=(18, 6))
g = sns.relplot(x='sketch_size', y='MAE', hue='algorithm', col='sparsity',
                data=results_df[results_df['dataset'] == 'generated'],
                kind='line', marker='o',
                facet_kws={'sharex': False, 'sharey': True, 'legend_out': True},
                col_wrap=3, height=5, aspect=1.2)

g.set(yscale='log', xlabel='Sketch Size', ylabel='MAE')  # 设置 y 轴为对数刻度, 并添加轴标签
g.fig.suptitle('Algorithm Performance on Generated Dataset (MAE)', fontsize=18, y=1.05)
g.fig.subplots_adjust(top=0.8, wspace=0.2, hspace=0.2)

# 调整图例位置
legend = g.axes[-1].legend(bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0.)
legend.get_frame().set_alpha(0.8)  # 设置图例背景的透明度

for i, ax in enumerate(g.axes):
    ax.set_title(f'Sparsity: {ax.get_title().split("=")[1].strip()}', fontsize=14)  # 调整子图标题的格式和字体大小

    if i % 3 != 0:  # 除了第一列的子图,其他子图隐藏y轴标签
        ax.set_ylabel('')

    ax.grid(True, linestyle='--', alpha=0.7)  # 添加网格线

plt.tight_layout()
plt.savefig('generated_dataset_mae.png', dpi=300, bbox_inches='tight')

# 绘制算法执行时间比较图
# 在绘制算法执行时间比较图之前,对数据进行分组和聚合
results_df_avg = results_df.groupby(['algorithm', 'dataset'], as_index=False)['avg_execution_time'].mean()

plt.figure(figsize=(12, 8))
sns.barplot(x='algorithm', y='avg_execution_time', hue='dataset', data=results_df_avg, ci=None, palette='muted')
plt.title('Algorithm Execution Time Comparison (Average across all sparsity and sketch_size settings)')
plt.xlabel('Algorithm')
plt.ylabel('Average Execution Time (s)')
plt.yscale('log')  # 使用对数刻度以更好地显示时间差异
plt.legend(title='Dataset', loc='upper left')
plt.tight_layout()
plt.savefig('algorithm_execution_time_comparison.png', dpi=300)


