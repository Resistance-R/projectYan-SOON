import pandas as pd
import numpy as np
from konlpy.tag import Mecab
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from filepath import out_data, text_indexes, one_hot_text_indexes

"""
단어사전 생선
"""

Out_Data = out_data
Text_Indexes = text_indexes

max_len = 600
num_classes = 3

def preprocess_text(df):
    """데이터 전처리"""
    mecab = Mecab()

    QList = []
    AList = []

    # 합친 데이터 불러오기
    df = pd.read_csv(df)

    # NaN 값 제거
    df = df.dropna(subset=['Q', 'A'])

    # Q열 데이터를 저장
    df['Q_morphs'] = df['Q'].apply(lambda x: mecab.morphs(x))
    QList.extend(df['Q_morphs'])

    # A열 데이터를 저장
    df['A_morphs'] = df['A'].apply(lambda x: mecab.morphs(x))
    AList.extend(df['A_morphs'])

    # 단어 사전 생성
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(QList + AList)

    w2i = tokenizer.word_index

    # <UNK> 토큰을 추가
    w2i['<UNK>'] = len(w2i) + 1

    word_index_df = pd.DataFrame(list(w2i.items()), columns=['word', 'index'])
    word_index_df.to_csv(Text_Indexes, index = False, encoding = 'utf-8')

    # 단어 사전 반환
    return word_index_df

def one_hot_dictionary(df, tokenizer):
    Y_train = np.array(df["A_padded"].tolist())
    Y_train_adjusted = (Y_train - np.min(Y_train)) / (np.max(Y_train) - np.min(Y_train)) * (num_classes - 1)
    Y_train_one_hot = to_categorical(Y_train_adjusted, num_classes=num_classes)

    # 새로운 토크나이저 생성 및 단어 사전 구축
    new_tokenizer = Tokenizer()
    new_tokenizer.fit_on_texts(df["A_morphs"].tolist())

    # 단어 사전 반환
    w2i_one_hot = new_tokenizer.word_index

    # <UNK> 토큰을 추가
    w2i_one_hot['<UNK>'] = len(w2i_one_hot) + 1

    word_index_df = pd.DataFrame(list(w2i_one_hot.items()), columns = ['word', 'index'])
    word_index_df.to_csv(one_hot_text_indexes, index = False, encoding = 'utf-8')

    return w2i_one_hot, Y_train_one_hot


def load_word_index(dictionary):
    """단어 사전 로드"""
    df = pd.read_csv(dictionary)
    word_index_dict = dict(zip(df['word'], df['index']))
    return word_index_dict