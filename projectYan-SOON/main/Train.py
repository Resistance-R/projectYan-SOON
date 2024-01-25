import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from Model import TransformerBlock, TokenAndPositionEmbedding


"""
모델 학습 코드
"""

def create_transformer_model(vocab_size, max_len, embed_dim, num_heads, ff_dim, dropout_rate, num_transformer_blocks, mlp_units, mlp_dropout):
    inputs = tf.keras.Input(shape=(max_len,))
    embedding_layer = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate) for _ in range(num_transformer_blocks)]
    for transformer_block in transformer_blocks:
        x = transformer_block(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(mlp_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(mlp_dropout)(x)
    outputs = tf.keras.layers.Dense(vocab_size, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def train_transformer_model(model, train_data, val_data, num_epochs, learning_rate):
    model.compile(optimizer=Adam(learning_rate), loss=SparseCategoricalCrossentropy(), metrics=[SparseCategoricalAccuracy()])

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=num_epochs,
    )

    return history