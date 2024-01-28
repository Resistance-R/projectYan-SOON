import numpy as np
from Model import build_transformer_model
from Model import num_classes
from Data_preprocessing import load_data, preprocess_data
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from filepath import yanchan_data, web_data, out_data, test_data, model_file
from keras.callbacks import EarlyStopping

"""
메인
"""

#데이터 불러오기
Web_Data = web_data
Yanchan_Data = yanchan_data
Out_Data = out_data
Test_Data = test_data

load_data(Web_Data, Yanchan_Data, Out_Data)

Model_File = model_file

max_len = 600

#데이터 전처리
preprocessed_df, tokenizer = preprocess_data(Out_Data, max_len)

#훈련 데이터
X_train, Y_train = np.array(preprocessed_df["Q_padded"].tolist()), np.array(preprocessed_df["A_padded"].tolist())

#모델 불러오기
model = build_transformer_model(max_len, vocab_size=10000, embedding_dim=256, num_heads=4, ff_dim=4)

#모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

Y_train_adjusted = (Y_train - np.min(Y_train)) / (np.max(Y_train) - np.min(Y_train)) * (num_classes - 1)

#훈련 데이터의 레이블을 원-핫 인코딩
Y_train_one_hot = to_categorical(Y_train_adjusted, num_classes=num_classes)

#테스트 데이터 전처리
preprocessed_test_df, tokenizer = preprocess_data(Test_Data, max_len)

X_test, Y_test = np.array(preprocessed_test_df["Q_padded"].tolist()), np.array(preprocessed_test_df["A_padded"].tolist())

Y_test_adjusted = (Y_test - np.min(Y_test)) / (np.max(Y_test) - np.min(Y_test)) * (num_classes - 1)

#테스트 데이터의 레이블을 원-핫 인코딩
Y_test_one_hot = to_categorical(Y_test_adjusted, num_classes=num_classes)

#훈련 조기 종료
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

#모델 훈련
model.fit(X_train, Y_train_one_hot, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])


#모델 평가
test_loss, test_accuracy = model.evaluate(X_test, Y_test_one_hot)

#모델 저장
model.save(Model_File)