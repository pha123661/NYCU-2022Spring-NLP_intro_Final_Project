import xml.etree.ElementTree as ET
import pandas as pd


class InputExample(object):
    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def get_train_example():

    train_benchmark = pd.read_csv(
        'benchmark_homo_train.csv', index_col='text_id')

    train_examples = []

    tree = ET.parse('data_homo_train.xml')
    root = tree.getroot()

    for text in root:
        words = []
        labels = []

        pun_loc = train_benchmark.loc[text.attrib['id'], 'word_id']

        for word in text:

            words.append(word.text)

            if (word.attrib['id'] == pun_loc):
                labels.append('P')
            else:
                labels.append('O')

        train_examples.append((words, labels))

    return create_examples(train_examples, 'train')


def get_test_example_and_submission():

    test_examples = []
    submission = pd.DataFrame(columns=['text_id', 'word_id'])

    tree = ET.parse('data_homo_test.xml')
    root = tree.getroot()

    for text in root:
        words = []
        labels = []

        submission.loc[len(submission.index), 'text_id'] = text.attrib['id']

        # 直接把test的label全部設成'O'
        for word in text:
            words.append(word.text)
            labels.append('O')

        test_examples.append((words, labels))

    return create_examples(test_examples, 'test'), submission


def create_examples(lines, set_type):

    examples = []

    for i, (word, label) in enumerate(lines):
        guid = "%s-%s" % (set_type, i)
        text = ' '.join(word)
        label = label
        examples.append(InputExample(guid=guid, text=text, label=label))

    return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):

    label_map = {label: i for i, label in enumerate(label_list, 1)}
    features = []

    for example in examples:

        example_word_list = example.text.split()
        labels = example.label

        tokens = []

        for i, word in enumerate(example_word_list):
            token = tokenizer.tokenize(word)
            tokens.append(token[0])

        label_ids = []
        for i, token in enumerate(tokens):
            label_ids.append(label_map[labels[i]])

        tokens.insert(0, '[CLS]')
        tokens.append('[SEP]')
        label_ids.insert(0, label_map['[CLS]'])
        label_ids.append(label_map['[SEP]'])

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        input_ids += [0] * (max_seq_length - len(input_ids))
        input_mask += [0] * (max_seq_length - len(input_mask))
        label_ids += [0] * (max_seq_length - len(label_ids))
        segment_ids = [0] * max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_ids=label_ids))
    return features
