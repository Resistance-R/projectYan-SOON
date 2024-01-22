import pandas as pd
from konlpy.tag import Mecab

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

def preprocess_data(df):
    """데이터 전처리"""
    mecab = Mecab()

    #Q열 데이터를 저장
    df['Q_morphs'] = df['Q'].apply(mecab.morphs)

    #A열 데이터를 저장
    df['A_morphs'] = df['A'].apply(mecab.morphs)

    return df