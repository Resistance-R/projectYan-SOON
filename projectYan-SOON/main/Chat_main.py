import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Mecab
import re
from Model import TokenAndPositionEmbedding, TransformerBlock
from filepath import model_file
from Build_word_index import load_word_index, one_hot_dictionary, preprocess_text
from Model import model
from Data_preprocessing import preprocess_data
from filepath import text_indexes, out_data

"""
채팅 코드
"""
Text_Index = text_indexes
Out_Data = out_data

#모델 로드
with tf.keras.utils.custom_object_scope({'TokenAndPositionEmbedding': TokenAndPositionEmbedding, 'TransformerBlock': TransformerBlock}):
    loaded_model = load_model(model_file)

#레이블 리스트
label_list = ["Label 0", "Label 1", "Label 2"]

#단어 사전 로드
load_dictionary = load_word_index(Text_Index)

#최대 문장 길이
max_len = 600

#Y_one_hot 사전 불러오기
Y_preprocessed_df, Y_tokenizer = preprocess_data(Out_Data, max_len)

w2i_one_hot, Y_train_one_hot = one_hot_dictionary(Y_preprocessed_df, Y_tokenizer)

#입력 데이터 전처리
def text_preprocess(text, w2i):
    mecab = Mecab()
    text = re.sub("[^가-힣]", "", text)
    malist = mecab.pos(text)

    encoded_malist = [w2i.get(w, w2i['<UNK>']) for w, _ in malist]

    return encoded_malist

def generate_response(prompt, word_index, max_len=max_len):
    #텍스트 정제, 벡터로 변환
    preprocessed_prompt = text_preprocess(prompt, word_index)

    #패딩
    padded_prompt = pad_sequences([preprocessed_prompt], maxlen=max_len, padding='post', truncating='post')

    #예측
    output_tokens = loaded_model.predict(padded_prompt)

    #예측 결과 해석
    generated_response = []

    for timestep in range(max_len):
        #시간 단계마다 최대 확률을 갖는 토큰의 인덱스
        predicted_index = np.argmax(output_tokens[0, timestep, :])

        #종료 토큰이 나오면 종료
        if predicted_index == 0:
            break

        #예측된 단어 확인
        predicted_word = [word for word, index in word_index.items() if index == predicted_index][0]
        generated_response.append(predicted_word)

    return " ".join(generated_response)

def chat():
    print("말을 걸어보세요. 종료를 원할 시 '!exit'를 입력하세요.")

    while True:
        user_input = input("당신: ")

        if user_input == "!exit":
            print("채팅 종료됨.")
            break

        bot_response = generate_response(user_input, load_dictionary)

        print("얀순: {}".format(bot_response))

if __name__ == "__main__":
    load_dictionary

    chat()