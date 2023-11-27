###########Library Load#############
import tensorflow as tf
from keras.layers import Embedding, LSTM, Dense, Dropout, Attention
###########Model Arc#############

class Encoder(tf.keras.Model):
    def __init__(self, units, vocab_size, embedding_dim, time_steps):
        super(Encoder, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim, input_length=time_steps, name='Embedding')
        self.dropout = Dropout(0.2, name='Dropout')
        # (attention) return_sequences=True 추가
        self.LSTM = LSTM(units, return_state=True, return_sequences=True, name='LSTM')
        self.vocab_size = vocab_size
    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.dropout(x)
        x, hidden_state, cell_state = self.LSTM(x)
        # (attention) x return 추가
        return x, [hidden_state, cell_state]

class Decoder(tf.keras.Model):
    def __init__(self, units, vocab_size, embedding_dim, time_steps):
        super(Decoder, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim, input_length=time_steps, name='Embedding')
        self.dropout = Dropout(0.2, name='Dropout')
        self.LSTM = LSTM(units,
                         return_state=True,
                         return_sequences=True,
                         name='LSTM'
                        )
        self.attention = Attention(name='Attention')
        self.dense = Dense(vocab_size, activation='softmax', name='Dense')

    def call(self, inputs, initial_state):
        # (attention) encoder_inputs 추가
        encoder_inputs, decoder_inputs = inputs
        x = self.embedding(decoder_inputs)
        x = self.dropout(x)
        x, hidden_state, cell_state = self.LSTM(x, initial_state=initial_state)

        # (attention) key_value, attention_matrix 추가
        # 이전 hidden_state의 값을 concat으로 만들어 vector를 생성합니다.
        key_value = tf.concat([initial_state[0][:, tf.newaxis, :], x[:, :-1, :]], axis=1)
        # 이전 hidden_state의 값을 concat으로 만든 vector와 encoder에서 나온 출력 값들로 attention을 구합니다.
        attention_matrix = self.attention([key_value, encoder_inputs])
        # 위에서 구한 attention_matrix와 decoder의 출력 값을 concat 합니다.
        x = tf.concat([x, attention_matrix], axis=-1)

        x = self.dense(x)
        return x, hidden_state, cell_state

class Seq2Seq(tf.keras.Model):
    def __init__(self, units, vocab_size, embedding_dim, time_steps, start_token, end_token):
        super(Seq2Seq, self).__init__()
        self.start_token = start_token
        self.end_token = end_token
        self.time_steps = time_steps
        self.vocab_size = vocab_size
        self.embedding_dim=embedding_dim
        self.units = units

        self.encoder = Encoder(self.units, self.vocab_size, self.embedding_dim, self.time_steps)
        self.decoder = Decoder(self.units, self.vocab_size, self.embedding_dim, self.time_steps)


    def call(self, inputs, training=True):
        if training:
            encoder_inputs, decoder_inputs = inputs
            # (attention) encoder 출력 값 수정
            encoder_outputs, context_vector = self.encoder(encoder_inputs)
            # (attention) decoder 입력 값 수정
            decoder_outputs, _, _ = self.decoder((encoder_outputs, decoder_inputs), initial_state=context_vector)
            return decoder_outputs
        else:
            x = inputs
            # (attention) encoder 출력 값 수정
            encoder_outputs, context_vector = self.encoder(x)
            target_seq = tf.constant([[self.start_token]], dtype=tf.float32)
            results = tf.TensorArray(tf.int32, self.time_steps)

            for i in tf.range(self.time_steps):
                decoder_output, decoder_hidden, decoder_cell = self.decoder((encoder_outputs, target_seq), initial_state=context_vector)
                decoder_output = tf.cast(tf.argmax(decoder_output, axis=-1), dtype=tf.int32)
                decoder_output = tf.reshape(decoder_output, shape=(1, 1))
                results = results.write(i, decoder_output)

                if decoder_output == self.end_token:
                    break

                target_seq = decoder_output
                context_vector = [decoder_hidden, decoder_cell]

            return tf.reshape(results.stack(), shape=(1, self.time_steps))
