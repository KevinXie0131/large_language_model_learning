import pandas as pd

df = pd.read_parquet('train-00000-of-00001.parquet')

print(df.info())
print('\n--- First few rows ---')
print(df.head())

print('\n--- Summary of annotation column ---')
print('Value counts:')
print(df['annotation'].value_counts())
print('\nUnique values:', df['annotation'].nunique())
print('\nSample values:')
print(df['annotation'].head(10).tolist())

print('\n--- Class Distribution of annotation column ---')
import matplotlib.pyplot as plt

annotation_counts = df['annotation'].value_counts()
annotation_percent = df['annotation'].value_counts(normalize=True) * 100

print('Class Distribution:')
print(annotation_counts)
print('\nPercentage by class:')
print(annotation_percent)
print('\nTotal classes:', df['annotation'].nunique())
print('Total samples:', len(df))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart
colors = plt.cm.Set3(range(len(annotation_counts)))
bars = axes[0].bar(range(len(annotation_counts)), annotation_counts.values, color=colors)
axes[0].set_xticks(range(len(annotation_counts)))
axes[0].set_xticklabels(annotation_counts.index if annotation_counts.index[0] != '' else ['Empty'], rotation=45, ha='right')
axes[0].set_title('Class Distribution of Annotation (Bar Chart)')
axes[0].set_xlabel('Annotation Class')
axes[0].set_ylabel('Count')
for bar, count in zip(bars, annotation_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(count), ha='center', va='bottom')

# Pie chart
labels = annotation_counts.index if annotation_counts.index[0] != '' else ['Empty']
axes[1].pie(annotation_counts.values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
axes[1].set_title('Class Distribution of Annotation (Pie Chart)')

plt.tight_layout()
plt.savefig('annotation_class_distribution.png', dpi=150)
plt.show()
print('\nChart saved as annotation_class_distribution.png')

