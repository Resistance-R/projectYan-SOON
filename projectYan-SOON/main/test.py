import numpy as np
from Model import max_len
from Data_preprocessing import preprocess_data
from filepath import test_data
from keras.utils import to_categorical
from keras.models import load_model
from Model import TokenAndPositionEmbedding, TransformerBlock
import tensorflow as tf

with tf.keras.utils.custom_object_scope({'TokenAndPositionEmbedding': TokenAndPositionEmbedding, 'TransformerBlock': TransformerBlock}):
    loaded_model = load_model("projectYan-SOON/main/transformer_model.h5")

num_classes = 3

Test_Data = test_data

preprocessed_test_df, _ = preprocess_data(Test_Data, max_len)
X_test = np.array(preprocessed_test_df["Q_padded"].tolist())

Y_test = np.array(preprocessed_test_df["A_padded"].tolist())
Y_test_adjusted = (Y_test - np.min(Y_test)) / (np.max(Y_test) - np.min(Y_test)) * (num_classes - 1)
Y_test_one_hot = to_categorical(Y_test_adjusted, num_classes=num_classes)

# 평가 함수를 사용하여 테스트 데이터에 대한 성능을 평가합니다.
test_loss, test_accuracy = loaded_model.evaluate(X_test, Y_test_one_hot)
print("테스트 손실:", test_loss)
print("테스트 정확도:", test_accuracy)