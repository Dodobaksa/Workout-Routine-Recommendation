###########Library Load#############
from keras.preprocessing.sequence import pad_sequences
import numpy as np
#############Functions###############
def make_question(sentence,maxlen,tokenizer):
   question_sequence = tokenizer.texts_to_sequences([sentence])
   question_padded = pad_sequences(question_sequence, maxlen=maxlen, truncating='post', padding='post')
   return question_padded
def make_prediction(model,question_inputs):
   results = model(inputs=question_inputs, training=False)
   # 변환된 인덱스를 문장으로 변환
   results = np.asarray(results).reshape(-1)
   return results
def convert_index_to_text(indexs, end_token,tokenizer):
    sentence = []
    # 모든 문장에 대해서 반복
    for index in indexs:
        if index == end_token:
            # 끝 단어이므로 예측 중비
            break;
        # 사전에 존재하는 단어의 경우 단어 추가
        if index > 0 and tokenizer.index_word[index] is not None:
            sentence.append(tokenizer.index_word[index])
        else:
        # 사전에 없는 인덱스면 빈 문자열 추가
            sentence.append('')
    return sentence
#############Make Prediction###############
class make_result:
    def __init__(self,model,tokenizer,maxlen):
        self.model= model
        self.tokenizer= tokenizer
        self.maxlen = maxlen
    def make_seq(self,question):
        question = [str(x) for x in question]
        question_inputs = make_question(question,self.maxlen,self.tokenizer)
        results = make_prediction(self.model, question_inputs)
        results = convert_index_to_text(results, 3,self.tokenizer)
        return results

