import pandas as pd
import re
import xml.etree.ElementTree as ET


def parse_label(word_id):
    '''
    Example:
        word_id = 'hom_17_8'
        rst = ['17', '8']
    -> returns 8
    '''
    pattern = '\d+'
    rst = re.findall(pattern, word_id)

    return int(rst[1])


def load_dataset(path, label=None, sentence_id=True):
    '''
    Parse xml into pandas dataframe
    Example:
        df['hom_1'].sentence = List['str'] # represents each word
    Optional:
        df['hom_1'].sentence_id = 'hom_1' # keep sentence_id for numpy array
        df['hom_1'].word_id = 11 # 11th word is the label (index starts from 0)
        df['hom_1'].target_word = 'sweat'
    '''
    tree = ET.parse(path)
    root = tree.getroot()
    df = {}
    for id in root:
        df[id.attrib['id']] = []
        for content in id:
            df[id.attrib['id']].append(content.text)

    df = pd.Series(df, dtype=object)
    df = pd.DataFrame(df, index=df.index, columns=['sentence'])

    # save sentence_id as a column
    if sentence_id:
        df['sentence_id'] = df.index

    if label is not None:
        labels = pd.read_csv(label, index_col='text_id')
        # subtract 1 since index starts from 0
        labels['word_id'] = labels['word_id'].map(parse_label) - 1
        df = pd.concat([df, labels], axis=1)
        df['target_word'] = df.apply(
            lambda series: series['sentence'][series['word_id']], axis=1)

    print("########################")
    print(f"Parsed {path}\nPreview:")
    print(df.head())
    print("########################")
    return df


def main():
    df = load_dataset(r"data\data_homo_train.xml",
                      label=r'data\benchmark_homo_train.csv',
                      sentence_id=False)
    load_dataset(r'data\data_homo_test.xml')


if __name__ == '__main__':
    main()
