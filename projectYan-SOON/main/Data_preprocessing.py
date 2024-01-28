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

def preprocess_data(df, max_len):
    """데이터 전처리"""
    mecab = Mecab()

    QList = []
    AList = []

    #합친 데이터 불러오기
    df = pd.read_csv(df)

    #NaN 값 제거
    df = df.dropna(subset=['Q', 'A'])

    #Q열 데이터를 저장
    df['Q_morphs'] = df['Q'].apply(lambda x: mecab.morphs(x))
    QList.extend(df['Q_morphs'])

    #A열 데이터를 저장
    df['A_morphs'] = df['A'].apply(lambda x: mecab.morphs(x))
    AList.extend(df['A_morphs'])

    #단어 사전 생성
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(QList + AList)

    w2i = tokenizer.word_index

    #<UNK> 토큰을 추가
    w2i['<UNK>'] = len(w2i) + 1

    #형태소를 정수로 인코딩
    df["Q_encoded"] = df["Q_morphs"].apply(lambda x: [w2i.get(w, w2i['<UNK>']) for w in x])
    df["A_encoded"] = df["A_morphs"].apply(lambda x: [w2i.get(w, w2i['<UNK>']) for w in x])

    #가장 긴 문장의 길이를 계산
    max_len = max(max(len(item) for item in df["Q_encoded"]), max(len(item) for item in df["A_encoded"]))

    #패딩을 적용
    Q_padded = pad_sequences(df["Q_encoded"].tolist(), maxlen=max_len, padding='post')
    A_padded = pad_sequences(df["A_encoded"].tolist(), maxlen=max_len, padding='post')

    #추가할 열의 이름을 설정하고 데이터를 DataFrame에 추가
    df["Q_padded"] = Q_padded.tolist()
    df["A_padded"] = A_padded.tolist()

    return df, tokenizer