###########Library Load#############
import pandas as pd
import os
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import random

###########Functions#############
def calculate_bmi(df):
    df['BMI'] = (df['weight'] / pow(df['height'],2)) *100
    df['age*bmi'] = df['age']*df['BMI']
    df.dropna(subset='seq',inplace=True)
    df.reset_index(drop=True,inplace=True)
    return df
def first_final_transform(df):
    pre_idx=[]
    for idx,v in enumerate(df['seq']):
        if '준비운동' in v and '마무리운동' in v and '본운동' in v:
            pre_idx.append(idx)
    sliced_df2 = df.loc[df.index.isin(pre_idx)]
    sliced_df2.reset_index(drop=True,inplace=True)
    return sliced_df2
def split_seq(df):
    df['3part_seq'] = [x.split('/') for x in df['seq']]
    sliced_df3 = df[['age','height','weight','BMI','age*bmi','sex','3part_seq']]
    sliced_df3['sex'] = [ 1 if x=='M'else 0 for x in sliced_df3['sex']]
    sliced_df3['divide_seq']=[x.replace('준비운동:','').replace('본운동:','').replace('마무리운동:','').replace('/','').split('  ') for x in df['seq']]
    sliced_df3['first_seq'] = [x[0].replace('준비운동:',"").split(',') for x in sliced_df3['3part_seq']]
    return sliced_df3
def item_reorder(input_list, b):
    if b < 0 or b > 1:
        raise ValueError("Invalid 'b' value")
    r = random.randint(0, len(input_list) - 1)  # 무작위로 'r' 선택
    end_index = r + int(len(input_list) * b)

    shuffled_portion = input_list[r:end_index]
    random.seed(42)
    random.shuffle(shuffled_portion)
    out_list = input_list[:r] + shuffled_portion + input_list[end_index:]
    return out_list
def be_refined(df):
    df['first'] = [x[0].replace('준비운동:',"").split(',') for x in df['3part_seq']]
    df['second'] = [x[1].replace('본운동:',"").split(',') for x in df['3part_seq']]
    df['third'] = [x[2].replace('마무리운동:',"").split(',') for x in df['3part_seq']]
    df.drop(['3part_seq','divide_seq','first_seq'],axis=1,inplace=True)
    candidate_df = df.copy()
    df['first'] = df['first'].apply(lambda x: item_reorder(x,0.9))
    df['second'] = df['second'].apply(lambda x: item_reorder(x,0.9))
    double_df = pd.concat([df,df],axis=0)
    double_df = double_df.dropna().reset_index(drop=True)
    double_df['after_first'] = double_df['second']+double_df['third']
    strip_first=[]
    strip_after=[]
    for idx in range(len(double_df)):
        strip_first.append([x.replace("  "," ").strip() for x in double_df['first'][idx]])
        strip_after.append([x.replace("  "," ").strip() for x in double_df['after_first'][idx]])
    double_df['strip_first'] = strip_first
    double_df['strip_after'] = strip_after
    double_df.drop(['first','second','third','after_first'],axis=1,inplace=True)
    double_df['first_p_first']=0
    for idx in tqdm(range(len(double_df))):
        double_df['first_p_first'][idx]=double_df['strip_first'][idx]+[double_df['strip_after'][idx][0]]
    return double_df
###########Get Start#############
class import_datasets:
    def __init__(self,folder_path):
        self.folder_path = folder_path
    def concat(self):
        # 해당 폴더 내의 모든 파일 목록 가져오기
        file_list = [f for f in os.listdir(self.folder_path) if f.endswith('.csv')]
        use_col = ['MESURE_AGE_CO','MESURE_IEM_001_VALUE','MESURE_IEM_002_VALUE','SEXDSTN_FLAG_CD','MVM_PRSCRPTN_CN']
        # 모든 CSV 파일을 하나의 DataFrame으로 병합
        dfs = [pd.read_csv(os.path.join(self.folder_path, file),usecols=use_col) for file in file_list]
        merged_df = pd.concat(dfs, ignore_index=True)
        return merged_df

class preprocessing:
    def __init__(self,df):
        self.df = df
    def prep(self):
        sliced_df = self.df[['MESURE_AGE_CO','MESURE_IEM_001_VALUE','MESURE_IEM_002_VALUE','SEXDSTN_FLAG_CD','MVM_PRSCRPTN_CN']]
        sliced_df.columns=['age','height','weight','sex','seq']
        sliced_df= calculate_bmi(sliced_df)
        sliced_df2 = first_final_transform(sliced_df)
        sliced_df3 = split_seq(sliced_df2)
        sliced_df3 = be_refined(sliced_df3)
        return sliced_df3

##############Split and Tokenizing###############
class split_n_tokenizing:
    def __init__(self,df,test_size,isshuffle:bool):
        self.df  = df
        self.test_size=test_size
        self.isshuffle=isshuffle
    def train_test_split(self):
        X_train,X_test = train_test_split(self.df,test_size=self.test_size,shuffle=self.isshuffle,random_state=42)
        survive_idx = X_train[['first_p_first']].drop_duplicates().index
        dedup_df = X_train.loc[survive_idx]
        question = dedup_df['first_p_first'].values.tolist()
        return dedup_df,question,X_test
    def tokenizing(self,dedup_df,question):
        answer_in = [['START']+x for x in dedup_df['strip_after']]
        answer_out = [x+['END'] for x in dedup_df['strip_after']]
        all_sentences = question + answer_in + answer_out
        max_src_len = max([len(line) for line in question])
        max_tar_len = max([len(line) for line in answer_out])
        print('source 문장의 최대 길이 :',max_src_len)
        print('target 문장의 최대 길이 :',max_tar_len)
        tokenizer = Tokenizer(filters='', lower=False, oov_token='<OOV>')
        tokenizer.fit_on_texts(all_sentences)
        question_sequence = tokenizer.texts_to_sequences(question)
        answer_in_sequence = tokenizer.texts_to_sequences(answer_in)
        answer_out_sequence = tokenizer.texts_to_sequences(answer_out)
        question_padded = pad_sequences(question_sequence, maxlen=max_src_len, truncating='post', padding='post')
        answer_in_padded = pad_sequences(answer_in_sequence, maxlen=max_tar_len, truncating='post', padding='post')
        answer_out_padded = pad_sequences(answer_out_sequence, maxlen=max_tar_len, truncating='post', padding='post')
        answer_in_one_hot = to_categorical(answer_in_padded)
        answer_out_one_hot = to_categorical(answer_out_padded,num_classes=713)
        print('START_TOKEN:',tokenizer.word_index['START'])
        print('END_TOKEN:',tokenizer.word_index['END'])
        print('VOCAB_SIZE:',len(tokenizer.word_index)+1)
        return question_padded,answer_in_padded,answer_out_one_hot,tokenizer