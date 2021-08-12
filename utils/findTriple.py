import json
import numpy as np
from keras.models import Model
from utils.bert4keras.snippets import open
from utils.bert4keras.tokenizers import Tokenizer
from utils.bert4keras.backend import keras, K, batch_gather
from utils.bert4keras.models import build_transformer_model
from utils.bert4keras.layers import LayerNormalization, Input, Lambda, Dense, Reshape

maxlen = 128

#模型路径
config_path = 'F:/demo/relation_extraction-Demo/utils/pretrained_model/albert_tiny_zh_google/albert_config_tiny_g.json'
checkpoint_path = 'F:/demo/relation_extraction-Demo/utils/pretrained_model/albert_tiny_zh_google/albert_model.ckpt'
dict_path = 'F:/demo/relation_extraction-Demo/utils/pretrained_model/albert_tiny_zh_google/vocab.txt'

predicate2id, id2predicate = {}, {}

with open('F:/demo/relation_extraction-Demo/utils/data/all_50_schemas', encoding='utf-8') as f:
    for l in f:
        l = json.loads(l)
        if l['predicate'] not in predicate2id:
            id2predicate[len(predicate2id)] = l['predicate']
            predicate2id[l['predicate']] = len(predicate2id)
num_labels = len(predicate2id)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def extrac_subject(inputs):
    """根据subject_ids从output中取出subject的向量表征
    """
    output, subject_ids = inputs
    subject_ids = K.cast(subject_ids, 'int32')
    start = batch_gather(output, subject_ids[:, :1])
    end = batch_gather(output, subject_ids[:, 1:])
    subject = K.concatenate([start, end], 2)
    return subject[:, 0]

def E2EModel(config_path, checkpoint_path, num_labels):
    # 补充输入
    subject_labels = Input(shape=(None, 2), name='Subject-Labels')
    subject_ids = Input(shape=(2,), name='Subject-Ids')
    object_labels = Input(shape=(None, num_labels, 2), name='Object-Labels')

    # 加载预训练模型
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model="albert",
        return_keras_model=False,
    )

    # 预测subject
    output = Dense(
        units=2, activation='sigmoid', kernel_initializer=bert.initializer
    )(bert.model.output)
    subject_preds = Lambda(lambda x: x ** 2)(output)

    subject_model = Model(bert.model.inputs, subject_preds)

    # 传入subject，预测object
    # 通过Conditional Layer Normalization将subject融入到object的预测中
    output = bert.model.layers[-2].get_output_at(-1)

    subject = Lambda(extrac_subject)([output, subject_ids])
    output = LayerNormalization(conditional=True)([output, subject])
    output = Dense(
        units=num_labels * 2,
        activation='sigmoid',
        kernel_initializer=bert.initializer
    )(output)
    output = Lambda(lambda x: x ** 4)(output)
    object_preds = Reshape((-1, num_labels, 2))(output)

    object_model = Model(bert.model.inputs + [subject_ids], object_preds)
    # 训练模型
    train_model = Model(
        bert.model.inputs + [subject_labels, subject_ids, object_labels],
        [subject_preds, object_preds]
    )
    return train_model, subject_model, object_model

train_model, subject_model, object_model = E2EModel(config_path, checkpoint_path, num_labels)

def extract_spoes(text):
    """抽取输入text所包含的三元组
    """
    tokens = tokenizer.tokenize(text, maxlen=maxlen)
    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
    # 抽取subject
    subject_preds = subject_model.predict([[token_ids], [segment_ids]])
    start = np.where(subject_preds[0, :, 0] > 0.6)[0]
    end = np.where(subject_preds[0, :, 1] > 0.5)[0]
    subjects = []
    for i in start:
        j = end[end >= i]
        if len(j) > 0:
            j = j[0]
            subjects.append((i, j))
    if subjects:
        spoes = []
        token_ids = np.repeat([token_ids], len(subjects), 0)
        segment_ids = np.repeat([segment_ids], len(subjects), 0)
        subjects = np.array(subjects)
        # 传入subject，抽取object和predicate
        object_preds = object_model.predict([token_ids, segment_ids, subjects])
        for subject, object_pred in zip(subjects, object_preds):
            start = np.where(object_pred[:, :, 0] > 0.6)
            end = np.where(object_pred[:, :, 1] > 0.5)
            for _start, predicate1 in zip(*start):
                for _end, predicate2 in zip(*end):
                    if _start <= _end and predicate1 == predicate2:
                        spoes.append((subject, predicate1, (_start, _end)))
                        break
        return [
            (tokenizer.decode(token_ids[0, s[0]:s[1] + 1], tokens[s[0]:s[1] + 1]),
             id2predicate[p],
             tokenizer.decode(token_ids[0, o[0]:o[1] + 1], tokens[o[0]:o[1] + 1]))
            for s, p, o in spoes
        ]
    else:
        return []

class SPO(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """

    def __init__(self, spo):
        self.spox = (
            tuple(tokenizer.tokenize(spo[0])),
            spo[1],
            tuple(tokenizer.tokenize(spo[2])),
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox

def getTriple(text):
    """展示函数
    """
    R = set([SPO(spo) for spo in extract_spoes(text)])
    spo_list = []
    for spo in R:
        spo_unit = {}
        spo_unit['object'] = spo[0]
        spo_unit['subject'] = spo[2]
        spo_unit['predicate'] = spo[1]
        spo_list.append(spo_unit)
    ieData = {"nodes": [], "links": []}
    existNodes = set()
    for spo in spo_list:
        obj = spo['object']
        sbj = spo['subject']
        if sbj not in existNodes:
            each_node = {}
            each_node['name'] = sbj
            ieData['nodes'].append(each_node)
            existNodes.add(sbj)
        if obj not in existNodes:
            each_node = {}
            each_node['name'] = obj
            ieData['nodes'].append(each_node)
            existNodes.add(obj)
        relation = spo['predicate']
        each_link = {}
        each_link['source'] = obj
        each_link['target'] = sbj
        each_link['value'] = relation
        ieData['links'].append(each_link)
    return ieData

if __name__ == '__main__':
    #测试
    train_model.load_weights('F:/demo/relation_extraction-Demo/utils/save/best_model.weights')
    text = '查尔斯·阿兰基斯（Charles Aránguiz），1989年4月17日出生于智利圣地亚哥，智利职业足球运动员，司职中场，效力于德国足球甲级联赛勒沃库森足球俱乐部。'
    ieData = getTriple(text)
    print(ieData)