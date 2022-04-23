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


def load_dataset(path, label=None, text_id_as_index=False):
    '''
    Parse xml into pandas dataframe
    Columns:
        text_id: Str, text id of this row
        text: List[Str], a list of each word in text
        word_id: Int, index of target word in sentence
        target_word: Str, target word (== text[word_id])

    Options:
        label: load label for training data
        text_id_as_index: set dataframe index as text_id
    '''
    tree = ET.parse(path)
    root = tree.getroot()
    df = {
        'text_id': [],
        'text': [],
    }

    for child in root:
        df['text_id'].append(child.attrib['id'])
        df['text'].append([content.text for content in child])

    df = pd.DataFrame(df)

    if label is not None:
        labels = pd.read_csv(label)
        # subtract 1 since index starts from 0
        labels['word_id'] = labels['word_id'].map(parse_label) - 1
        # merge data & label
        df = pd.merge(df, labels, how='inner')
        # extract target word from text
        df['target_word'] = df.apply(
            lambda series: series['text'][series['word_id']], axis=1)

    if text_id_as_index:
        # set text_id column as index
        df.set_index('text_id', inplace=True)

    print("########################")
    print(f"Parsed {path}\nPreview:")
    print(df.head())
    print("########################")
    return df


def main():
    train_df = load_dataset(r"data\data_homo_train.xml",
                            label=r'data\benchmark_homo_train.csv',
                            text_id_as_index=True)
    test_df = load_dataset(r'data\data_homo_test.xml')


if __name__ == '__main__':
    main()
