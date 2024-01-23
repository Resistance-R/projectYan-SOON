import pandas as pd
from konlpy.tag import Mecab
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

"""
데이터 전처리 관련 코드
"""

def load_data(web_data, yanchan_data, out_data):
    #깃허브 테이터와 얀챈 데이터를 불러온다.
    df1 = pd.read_csv(web_data)
    df2 = pd.read_csv(yanchan_data)

    #그 둘을 합친다.
    df_out = pd.concat([df1, df2], ignore_index = True)

    #데이터를 저장한다.
    df_out.to_csv(out_data, index = False)

    return df_out

def build_vocab(data):
    """단어 사전 생성"""
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)
    return tokenizer.word_index

def preprocess_data(df):
    """데이터 전처리"""
    mecab = Mecab()

    QList = []
    AList = []

    #Q열 데이터를 저장
    df['Q_morphs'] = df['Q'].apply(mecab.morphs)

    QList.extend(df['Q_morphs'])

    #A열 데이터를 저장
    df['A_morphs'] = df['A'].apply(mecab.morphs)

    AList.extend(df['A_morphs'])

    # 단어 사전 생성
    w2i = build_vocab(QList + AList)

    # 형태소를 정수로 인코딩
    df["Q_encoded"] = df["Q_morphs"].apply(lambda x: [w2i[w] for w in x])
    df["A_encoded"] = df["A_morphs"].apply(lambda x: [w2i[w] for w in x])

    # 가장 긴 문장의 길이를 계산
    QMax_len = max(len(item) for item in df["Q_encoded"])
    AMax_len = max(len(item) for item in df["A_encoded"])

    # 패딩을 적용
    Q_padded = pad_sequences(df["Q_encoded"].tolist(), maxlen=QMax_len, padding='post')
    A_padded = pad_sequences(df["A_encoded"].tolist(), maxlen=AMax_len, padding='post')

    # 추가할 열의 이름을 설정하고 데이터를 DataFrame에 추가
    df["Q_padded"] = Q_padded.tolist()
    df["A_padded"] = A_padded.tolist()

    #디버그
    # print("QMax_len:", QMax_len)
    # print("Q_padded:", Q_padded)

    # print("AMax_len:", AMax_len)
    # print("A_padded:", A_padded)

    return df