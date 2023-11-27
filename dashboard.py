#!/usr/bin/env python
# coding: utf-8
###########Library Load##########
import pandas as pd
import numpy as np
import pickle
import zipfile
import argparse
import streamlit as st
import warnings
from sklearn.preprocessing import MinMaxScaler
from seq2seq import Seq2Seq
from functions import most_similar_weight,calculate_bmr,calculate_daily_calories,calculate_daily_nutrient_intake
from postprocessing import make_result
from metrics import ap_at_k
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
###########Argparser#############
parser = argparse.ArgumentParser()
parser.add_argument('--weights_path',type=str, default='configuration/model_weights.ckpt')
args = parser.parse_args()
warnings.filterwarnings("ignore")
############Data load#############
@st.cache_data
def load_data():
    complete_df = pd.read_csv('complete_df.csv')
    return complete_df
@st.cache_data
def load_dedup():
    dedup_df = pd.read_csv('dedup_df.csv')
    return dedup_df
# 파일에서 딕셔너리 로드
@st.cache_data
def load_workout_type():
    with open('workout_ability.pkl', 'rb') as file:
        workout_type = pickle.load(file)
    return workout_type
@st.cache_data
def load_detail_ex():
    workout_df = pd.read_csv('workout.csv')
    return workout_df
def load_TOKENIZER():
    with open('configuration/tokenizer2.pkl', 'rb') as file:
        TOKENIZER = pickle.load(file)
    return TOKENIZER
############Functions#############
def set_model():
    my_model = Seq2Seq(256, 713, 100, 76, 2, 3)
    if args.weights_path:
        my_model.load_weights(args.weights_path)
    return  my_model
def extract_seq(my_model,TOKENIZER,INPUT_LEN,first_seq):
    init = make_result(my_model, TOKENIZER, INPUT_LEN)
    infered_seq = init.make_seq(first_seq)
    return infered_seq
def filter_ex_type(type):
    work_type = pd.DataFrame(type).fillna(0)
    workout_after = pd.concat([work_type,work_type],axis=0).reset_index(drop=True)
    return workout_after
def con_ex_info(df1,df2):
    add_info = pd.merge(df1['1순위'],df2,how='left',left_on='1순위',right_on='trng_nm')
    add_info2 = add_info.drop_duplicates(subset='1순위').drop(['trng_nm','trng_aim_nm'],axis=1)
    add_info2.columns = ['1순위','발달능력','운동부위','준비물']
    add_info2['발달능력'].fillna('-',inplace=True)
    add_info2['운동부위'].fillna('전신',inplace=True)
    add_info2['준비물'].fillna('없음',inplace=True)
    return add_info2
def get_url(workout_name:str):
    user_agent= '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.85 Safari/537.36'
    # 웹드라이버 option 설정 (생락가능)
    options = webdriver.ChromeOptions()
    options.add_argument('--ignore-certificate-errors')
    options.add_argument(user_agent)
    options.add_argument('--headless') 
    # 웹드라이버 설정
    # driver = webdriver.Chrome('/path/to/chromedriver')
    driver = webdriver.Chrome(options=options)
    # 유튜브 접속
    driver.get('https://www.youtube.com')

    # 유튜브 검색창에 키워드 입력
    search_box = driver.find_element(By.XPATH, '//input[@id="search"]')
    search_box.send_keys(f"체력100 {workout_name}")
    # 검색 버튼 클릭
    search_button = driver.find_element(By.XPATH, '//button[@id="search-icon-legacy"]')
    search_button.click()
    # 상위 동영상 찾기
    first_video = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'a#video-title')))
    # 첫 번째 동영상 클릭
    first_video.click()
    video_url = driver.current_url
    # 브라우저 닫기
    #driver.quit()
    return video_url
def make_warmup_df(recomm_first,proba):
    recomm_delete = [v[:-1] if len(v)>1 else v for v in recomm_first]
    probability = [x for x in np.round(proba / np.sum(proba)*100,2)]
    workout=pd.DataFrame(recomm_delete)
    workout2 = pd.concat([pd.DataFrame(probability),workout],axis=1)
    workout2.index=['1순위','2순위','3순위','4순위','5순위']
    workout2.columns=[['rating(%)']+[str(x)+'st운동' for x in range(1,len(workout2.columns))]][0]
    workout2.fillna('없음',inplace=True)
    workout2['select'] = [False for _ in range(len(workout2))]
    return workout2
def make_main_df(all_seq,detail_workout):
    result = list(dict.fromkeys(all_seq))
    after_recomm = pd.DataFrame(result).fillna('없음')
    after_recomm.columns=['1순위']
    after_recomm.index=[str(x)+'st운동' for x in range(1,len(after_recomm)+1)]
    detailed_ex = con_ex_info(after_recomm,detail_workout)
    detailed_ex.dropna(axis=0,inplace=True)
    detailed_ex['select'] = [False for _ in range(len(detailed_ex))]
    detailed_ex.index = [str(x)+'st운동' for x in range(1,len(detailed_ex)+1)]
    return detailed_ex
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity
# 유클리디안 거리 계산 함수
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)
def most_similar_weight_for_None(profile,df,k):
    mm=MinMaxScaler()
    df_re = df.copy()
    bmi = profile[2] / pow(profile[1],2)
    abi= profile[0]*bmi
    all = profile+[bmi]+[abi]
    euclidean_distances = 1 / df_re.iloc[:,:6].apply(lambda row: euclidean_distance(all, np.array(row)), axis=1)
    sim_scaled= (mm.fit_transform(euclidean_distances.values.reshape(-1,1))).reshape(1,-1)
    df_re['dist'] = sim_scaled[0]
    df_re = df_re.dropna().sort_values('dist',ascending=False)
    closest_indices = df_re['first_p_first'].values.tolist()[:k]
    proba = df_re['dist'].values.tolist()[:k]
    return closest_indices, proba
def most_similar_weight(profile,purpose,df,profile_w,candidate_w,k):
    mm= MinMaxScaler()
    df_re=df.copy()
    bmi = profile[2] / pow(profile[1],2)
    abi= profile[0]*bmi
    all = profile+[bmi]+[abi]
    up_idx = [['근력/근지구력','심폐지구력','민첩성/순발력','유연성','평형성','협응력'].index(x) for x in purpose]
    cadidates= [0,0,0,0,0,0]
    for v in up_idx:
        cadidates[v]=1
    # 각 행의 벡터와 주어진 벡터 사이의 유사도와 거리 계산
    cosine_similarities = df_re[['근력/근지구력','심폐지구력','민첩성/순발력','유연성','평형성','협응력']].apply(lambda row: cosine_similarity(cadidates, np.array(row)), axis=1)
    euclidean_distances = 1 / df_re[['age','height','weight','BMI','age*bmi','sex']].apply(lambda row: euclidean_distance(all, np.array(row)), axis=1)
    sim_scaled = (mm.fit_transform(euclidean_distances.values.reshape(-1,1))*profile_w).reshape(1,-1)+cosine_similarities.values*candidate_w
    df_re['dist'] = sim_scaled[0]
    df_re = df_re.dropna().sort_values('dist',ascending=False)
    closest_indices = df_re['first_p_first'].values.tolist()[:k]
    proba = df_re['dist'].values.tolist()[:k]
    return closest_indices, proba

def dedup_list(list):
    # 각 문자열의 겉에 있는 작은 따옴표 제거
    data = [x.strip("[]").replace("'", "") for x in list]
    # 리스트 내의 중첩 리스트를 단일 리스트로 풀기
    result = [x.split(", ") if ", " in x else [x] for x in data]
    return result
def longterm_routine(day,age,height,weight,gender,df,my_model,tokenizer,detail_workout):
    long_df = pd.DataFrame()
    recomm_first,_= most_similar_weight_for_None([age,height,weight,gender],df,day)
    for day in range(0,day):
        all_seq = extract_seq(my_model,tokenizer,26,recomm_first[day])
        daily_routine = pd.DataFrame(all_seq)
        daily_routine['일차']=f'{day+1}일차'
        long_df = pd.concat([long_df,daily_routine],axis=0)
    long_df_add = pd.merge(long_df,detail_workout,how='left',left_on=0,right_on='trng_nm').drop_duplicates(subset='trng_nm')
    long_df_add.set_index('일차',inplace=True)
    long_df_add.drop(['trng_nm','trng_aim_nm'],axis=1,inplace=True)
    long_df_add['ftns_fctr_nm'].fillna('-',inplace=True)
    long_df_add['trng_part_nm'].fillna('전신',inplace=True)
    long_df_add['tool_nm'].fillna('없음',inplace=True)
    long_df_add.columns=['운동이름','발달능력','운동부위','준비물']
    return long_df_add
###########Initialize#############
set_page = st.set_page_config(
    page_title="Workout Routine Recommendations",
    page_icon="💪",
    layout="wide",
)
# Store the initial value of widgets in session state
if "start_btn" not in st.session_state:
    st.session_state.start_btn = False
###########Load Configuration###########
complete_df = load_data()
complete_df['first_p_first'] = dedup_list(complete_df['first_p_first'])
dedup_df = load_dedup()
dedup_df.set_index('Unnamed: 0',inplace=True)
workout_type= load_workout_type()
workout_after = filter_ex_type(workout_type)
detail_workout=load_detail_ex()  
TOKENIZER=load_TOKENIZER()
my_model= set_model()
##############Side bar#############
def main():  
    st.sidebar.title('Configuration')
    st.sidebar.subheader('맞춤형 운동 루틴 추천을 위한 옵션',divider='red')
    age = st.sidebar.number_input("🙋🏻나이를 입력하세요.",  value=20,min_value=7, max_value=100)
    height = st.sidebar.number_input("🏃키를 입력하세요.",  value=170,min_value=140,max_value=350)
    weight = st.sidebar.number_input("🦶몸무게를 입력하세요.",  value=50,min_value=30,max_value=300)
    sex =st.sidebar.radio(
        "♂︎♀︎성별을 고르세요.",
        ['남자','여자'],
        horizontal=True
        )
    if sex=='남자':
        gender = 1
    elif sex=='여자':
        gender=0
    purpose =st.sidebar.multiselect(
        "🏂운동 목표을 고르세요",
        ['없음','근력/근지구력','심폐지구력','민첩성/순발력','유연성','평형성','협응력'],
        )

    activity =st.sidebar.radio(
        "🏊활동 수준을 고르세요.",
        ['집에만 있음','약간 활동적','적당히 활동적','매우 활동적'],
        horizontal=True
        )
    updown =st.sidebar.radio(
        "🏋️‍♀️희망 체중을 고르세요.",
        ['체중 감소','체중 유지','체중 증가'],
        horizontal=True
        )
    long_recomm =st.sidebar.radio(
        "🤾🏻‍♂️운동 루틴 추천 기간을 고르세요.",
        ['3일','7일','14일'],
        horizontal=True
        )
    st.sidebar.subheader('운동영상 검색',divider='red')
    workout_name = st.sidebar.text_input(
                "🎯운동이름을 입력하고 Enter를 누르세요 👇",
                label_visibility="visible",
                disabled=False,
                placeholder='줄넘기..동적 스트레칭..',
                )
    ################Body##################
    st.title('Workout Routine Recommendation')
    s_col1,s_col2,s_col3,s_col4,s_col5 = st.columns([1,1,1,1,1])
    with s_col1:
        bmr = calculate_bmr(gender, age, height, weight)
        st.metric(label='기초 대사량(BMR): ', value=f'{bmr} kcal')
    with s_col2:
        daily_cal = calculate_daily_calories(bmr, activity)
        st.metric(label='하루 권장 칼로리 섭취량(TTE): ', value=f'{daily_cal} kcal')
    with s_col3:
        neutrinuts = calculate_daily_nutrient_intake(daily_cal,updown)
        st.metric(label='하루 적정 탄수화물 섭취량: ', value=f'{np.round(neutrinuts[0],3)} g')
    with s_col4:
        st.metric(label='하루 적정 단백질 섭취량: ', value=f'{np.round(neutrinuts[1],3)} g')
    with s_col5:
        st.metric(label='하루 적정 지방 섭취량: ', value=f'{np.round(neutrinuts[2],3)} g')
    with st.container():
        con_df = pd.concat([complete_df.iloc[:,:9],workout_after],axis=1)
        new_criteria= con_df.loc[dedup_df.index]
        activity_des = st.expander('BMR,TTE,영양소 계산 방식에 대한 설명을 보려면 👉')
        with activity_des:
             with st.container():
                 st.write('● 기초대사량(Basal Metabolic Rate, BMR)은 안정된 상태에서 최소한의 에너지를 유지하기 위해 필요한 에너지 양을 말합니다. 쉽게 말해, 몸이 안정된 상태에서도 기능을 유지하기 위해 필요한 최소한의 칼로리 소비량입니다. 호흡, 혈액순환, 체온 조절 등의 기초적인 생리 작용을 수행하는 데 필요한 에너지 양을 나타냅니다.해리스-베네딕트 공식으로 계산되었습니다.')
                 st.write('● Total Energy Expenditure(TTE)는 일일 총 에너지 소비량을 나타냅니다. 우리가 하루 동안 소비하는 총 칼로리 양을 의미합니다. TTE는 기초대사량(BMR, Basal Metabolic Rate)과 신체활동에 의한 에너지 소비를 포함한 값으로, BMR에 활동 수준을 곱하여 계산됩니다.')
                 st.write('● 영양소 섭취량은 강남세브란스, 국민건강영양조사 자료 통해 사망률 낮은 영양소 섭취 비율 분석에 따른 하루 권장 탄수화물,단백질,지방의 비율로 5:2:3을 바탕으로 계산되었습니다.')
        see_data = st.expander('Raw data를 먼저 살펴보려면 👉')
        with see_data:
            st.dataframe(con_df.drop('first_p_first',axis=1).head(500).rename(columns={'strip_first':'warmup_excercise','strip_after':'main_excercise'}))
        st.snow()
        explanation = st.expander('추천 결과에 대한 설명을 보려면 👉')
        with explanation:
            with st.container():
                st.write("● 본 페이지는 준비 운동 5개와 본 운동 루틴 1개에 대한 추천 결과를 생성합니다. \
                         나이, 키 등 사용자 정보와 운동 목표에 따라 추천 결과가 달라집니다.")
                st.write("● 운동 목표 '없음'으로 체크할 시에는 사용자에게 일반적인 운동 루틴을 추천합니다.\
                         일반적인 운동 루틴 추천 결과는 사용자 정보에 기반합니다.")
                st.write("● raing(%)은 총합이 1이며 5개의 추천 결과에 대한 가중 점수입니다. \
                         중복되는 추천 결과는 중복되는 만큼 추천점수가 높음을 반영합니다.")
                st.write("● 맨 오른쪽에 체크박스를 클릭하면 추천 모델의 성능이 표시됩니다.\
                         추천 성능은 Online-test에 적합한 정밀도로 측정되었으며 정밀도에 대한 설명은 아래와 같습니다.")
            st.divider()
            st.text_area(
                "Precision@k (정밀도@k)",                                                
                "Precision@k는 상위 k개의 결과 중에서 실제로 정답인 비율을 의미합니다. "
                "예를 들어, 상위 5개의 결과를 가져왔을 때, 이 중 3개가 정답이라면 Precision@5는 0.6입니다. "
                "즉, 상위 k개의 결과 중에 실제로 정답인 것이 얼마나 포함되어 있는지를 나타내는 지표입니다."
                "본 페이지에서 Precision@k는 준비운동과 본 운동 모두를 측정대상으로 했습니다."
                )
            st.text_area(
                "Average Precision@k (평균 정밀도@k)",                                                
                "Average Precision@k는 상위 k개의 결과에 대한 Precision 값을 모두 구하고, 이를 정답이 있는 위치마다 평균을 내어 계산합니다."                        
                "이것은 검색 결과의 순서가 중요할 때 사용됩니다. 정답이 있는 위치에서의 Precision 값들을 평균내어 그 모델의 성능을 평가합니다. "
                "이는 모델이 얼마나 정확한 순서로 결과를 제시하는지에 대한 평가로 볼 수 있습니다."
                "본 페이지에서 Average Precision@k는 순서가 중요한 본 운동만을 측정대상으로 했습니다."
                )
    if purpose==['없음'] or purpose==[]:
        recomm_first,proba = most_similar_weight_for_None([age,height,weight,gender],con_df,5)
    else:
        recomm_first,proba = most_similar_weight([age,height,weight,gender],purpose,new_criteria,0.01,0.99,5)
    workout2 = make_warmup_df(recomm_first,proba)
    data_as_csv2= workout2.to_csv(index=False).encode("utf-8")
    st.subheader('준비운동(Warmup) 추천 결과', divider='red')
    st.download_button(
            label="Download Warmup Excercise as CSV",
            data=data_as_csv2,
            file_name='warmup_list.csv',
            mime='text/csv',
        )
    if "warmup" not in st.session_state:
        st.session_state.warmup=workout2
    warmup_df = st.data_editor(workout2,                            
                   column_config={
                        "select": st.column_config.CheckboxColumn(
                            "What exercise will you do?",
                            help="Select your **warmup** exercise",
                            default=False,
                        )
                    })
    all_seq = extract_seq(my_model,TOKENIZER,26,recomm_first[0])
    detailed_ex = make_main_df(all_seq,detail_workout)
    detailed_ex.rename(columns={'1순위':'운동이름'},inplace=True)
    # 이전에 선택한 체크 상태를 세션 상태에 저장
    data_as_csv= detailed_ex.to_csv(index=False).encode("utf-8")
    st.error('화면이 어두워져도, check box를 누르세요. 추천 성능이 표시됩니다.' ,icon='❣')
    st.subheader('본 운동(Main) 추천 결과', divider='red')
    st.download_button(
        label="Download Main Excercise as CSV",
        data=data_as_csv,
        file_name='excercise_list.csv',
        mime='text/csv',
    )
    if "main_ex" not in st.session_state:
        st.session_state.main_ex=detailed_ex
    m1, m2,m3= st.columns([7,1.5,1.5])
    with m1:
        main_df = st.data_editor(
                       detailed_ex,
                       column_config={
                           "select": st.column_config.CheckboxColumn(
                               "What exercise will you do?",
                               help="Select your **today`s** workout",
                               default=False,
                           )
                       },
                       disabled=["1순위"],
                       hide_index=False
                   )
    if main_df['select'].any() or warmup_df['select'].any():
        precision = (len(list(main_df[main_df['select']].index))+len(list(warmup_df[warmup_df['select']].index))) / (len(main_df)+len(warmup_df))
        liked_item = main_df[main_df['select']==True]['운동이름'].values.tolist()
        recomm_item = main_df['운동이름'].values.tolist()
        ap = ap_at_k(liked_item,recomm_item,len(main_df))
        m2.metric(label=f'추천 성능1(Precision@{len(main_df)+len(warmup_df)})', value=np.round(precision,3))
        m3.metric(label=f"추천 성능2(Average precision@{len(main_df)})", value=np.round(ap,3))
    if long_recomm=='14일':
        day = int(long_recomm[:2])
    else:
        day = int(long_recomm[0])
    long_df = longterm_routine(day,age,height,weight,gender,new_criteria,my_model,TOKENIZER,detail_workout)
    data_as_csv3= long_df.to_csv(index=False).encode("utf-8")
    st.subheader(f'{long_recomm} 운동 루틴 추천 결과', divider='red')
    st.download_button(
        label=f"Download {day}days Excercise Routine as CSV",
        data=data_as_csv3,
        file_name=f'{day}days excercise_list.csv',
        mime='text/csv',
    )
    if "long_ex" not in st.session_state:
        st.session_state.long_ex=long_df
    st.table(long_df)
    # 입력값이 있을 경우 동작
    if workout_name:
        video_url = get_url(workout_name)
        st.sidebar.write(f"URL: {video_url}")

###########Implement############
if __name__ == "__main__":
    main()
    