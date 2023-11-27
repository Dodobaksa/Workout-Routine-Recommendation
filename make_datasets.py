###########Library Load#############
import numpy as np
import argparse
import pickle
import pandas as pd
from tqdm import tqdm
from preprocessing import import_datasets,preprocessing,split_n_tokenizing
###########Argparser#############
parser = argparse.ArgumentParser()
parser.add_argument('-file_path',type=str, default='C:/Users/gram/Desktop/체육진흥공단_공모전/dataset')
parser.add_argument('-workout_path',type=str, default='dashboard/workout.csv')
parser.add_argument('-test_size',type=float, default=0.3)
parser.add_argument('-isshuffle',type=bool, default=True)
args = parser.parse_args()
############Implements##############
def main():
    init = import_datasets(args.file_path)
    merged_df = init.concat()
    set_data  = preprocessing(merged_df)
    complete_df = set_data.prep()
    complete_df.to_csv('complete_df.csv',index=False)
    init2= split_n_tokenizing(complete_df,args.test_size,args.isshuffle)
    dedup_df,question,X_test = init2.train_test_split()
    X_test.to_csv('X_test.csv',index=False)
    question_padded,answer_in_padded,answer_out_one_hot,tokenizer = init2.tokenizing(dedup_df,question)
    # 객체를 파일에 저장
    with open('question_padded.pkl', 'wb') as file:
        pickle.dump(question_padded, file)
    with open('answer_in_padded.pkl', 'wb') as file:
        pickle.dump(answer_in_padded, file)
    with open('answer_out_one_hot.pkl', 'wb') as file:
        pickle.dump(answer_out_one_hot, file)
    with open('tokenizer.pkl', 'wb') as file:
        pickle.dump(tokenizer, file)
    ########Open Api에서 가져온 운동종류###########
    workout_df = pd.read_csv(args.workout_path)
    ex_ratio=[]
    for idx in tqdm(range(len(complete_df))):
        ratio = pd.merge(pd.DataFrame(complete_df['after_first'][idx]),workout_df,how='left',left_on=0,right_on='trng_nm')['ftns_fctr_nm'].value_counts()
        ex_ratio.append(dict(ratio / np.sum(ratio)))
    # 파일에 딕셔너리 저장
    with open('work_out_ratio.pkl', 'wb') as file:
        pickle.dump(ex_ratio, file)
###########Implement############
if __name__ == "__main__":
    main()