###########Library Load#############
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from seq2seq import Seq2Seq
##############Forward###############
class train:
    def __init__(self,BATCH_SIZE,EMBEDDING_DIM,UNITS,EPOCHS,WEIGHTS_PATH):
        self.batch_size = BATCH_SIZE
        self.embedding_dim = EMBEDDING_DIM
        self.units = UNITS
        self.epochs = EPOCHS
        self.weights_path=WEIGHTS_PATH
    def train(self, question_padded,answer_in_padded,answer_out_one_hot,TOKENIZER,load_model:bool):
        TIME_STEPS = len(answer_out_one_hot[0])
        VOCAB_SIZE = len(TOKENIZER.word_index) + 1
        DATA_LENGTH = len(question_padded)
        INPUT_LEN = question_padded[0].shape[0]
        earlystopping = EarlyStopping(monitor='loss',  # 모니터 기준 설정 (val loss)
                                         patience=15
                                        ,verbose=0, restore_best_weights=True) # 15회 Epoch동안 개선되지 않는다면 종료
        model_path = f'./seq2seq'
        mc = ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_loss', mode='auto', verbose=0)
        seq2seq = Seq2Seq(self.units, VOCAB_SIZE, self.embedding_dim, TIME_STEPS, 2, 3)
        seq2seq.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        with tf.device("/device:GPU:0"):
            seq2seq.fit([question_padded, answer_in_padded],
                answer_out_one_hot,
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[earlystopping, mc],
                #validation_data=([answer_in_val,answer_out_val],X_val_q),
               )
            if load_model:
                seq2seq.load_weights(self.weights_path)
                return TOKENIZER,seq2seq,INPUT_LEN
            else:
                load_model = seq2seq
                return TOKENIZER,load_model,INPUT_LEN
    def export_weights(self,model):
        # 가중치를 HDF5 파일로 저장
        model.save_weights(f"model_weights.ckpt")




