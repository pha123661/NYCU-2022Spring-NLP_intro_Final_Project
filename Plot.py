import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Load_data import load_dataset
import spacy

nlp = spacy.load("en_core_web_sm")

sns.set(rc={'figure.figsize': (11.7, 8.27)})


df = load_dataset(r"code\data_homo_train.xml",
                  label=r'code\benchmark_homo_train.csv',
                  text_id_as_index=True)


sns.countplot(
    x=df['word_id'] + 1,
    order=list(range(1, 16)),
).set(
    title='Pun Location Distribution',
    xlabel='Pun Location',
    ylabel='Count'
)

plt.axvline(x=df['word_id'].mean() + 1, linestyle='--')

plt.savefig('location dis.png')

plt.figure()

sns.histplot(
    x=df['word_id'] / df['text'].apply(len),
    kde=True,
).set(
    title='Pun Location Distribution (%)',
    xlabel='Pun Location (%)',
    ylabel='Count'
)

plt.savefig("location percentage.png")

plt.figure()

df['doc'] = df['text'].apply(lambda x: " ".join(x)).apply(nlp)
df['POS'] = df.apply(lambda s: str(s['doc'][s['word_id']].pos_), axis=1)

sns.countplot(x=df['POS']).set(
    title='Pun POS Tag Distribution', xlabel='POS tag', ylabel='Count')
plt.savefig('POS dis.png')
