import tensorflow as tf
from Model import build_transformer_model  # 트랜스포머 모델을 정의한 파일
from Data_preprocessing import load_data, preprocess_data  # 데이터 전처리 함수를 정의한 파일
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from filepath import yanchan_data, web_data, out_data

"""
메인
"""

#데이터 불러오기
Yanchan_Data = yanchan_data
Web_Data = web_data
Out_Data = out_data