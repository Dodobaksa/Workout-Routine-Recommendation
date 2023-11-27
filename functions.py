###########Library Load#############
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
##########Functions#############
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity
# 유클리디안 거리 계산 함수
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)
def most_similar_weight(profile,purpose,con_df,profile_w,candidate_w):
    mm=MinMaxScaler()
    if purpose==['없음'] or purpose==[]:
        bmi = profile[2] / pow(profile[1],2)
        abi= profile[0]*bmi
        all = profile+[bmi]+[abi]
        euclidean_distances = con_df.iloc[:,:6].apply(lambda row: euclidean_distance(all, np.array(row)), axis=1)
        sim_scaled= (mm.fit_transform(euclidean_distances.values.reshape(-1,1))).reshape(1,-1)
        closest_indices = np.argsort(sim_scaled[0].tolist())[:5]
        proba= (sorted(sim_scaled[0].tolist(),reverse=True))[:5]
    else:
        bmi = profile[2] / pow(profile[1],2)
        abi= profile[0]*bmi
        all = profile+[bmi]+[abi]
        up_idx = [['근력/근지구력','심폐지구력','민첩성/순발력','유연성','평형성','협응력'].index(x) for x in purpose]
        cadidates= [0,0,0,0,0,0]
        for v in up_idx:
            cadidates[v]=1
        # 각 행의 벡터와 주어진 벡터 사이의 유사도와 거리 계산
        cosine_similarities = con_df.iloc[:,8:].apply(lambda row: cosine_similarity(cadidates, np.array(row)), axis=1)
        euclidean_distances = con_df.iloc[:,:6].apply(lambda row: euclidean_distance(all, np.array(row)), axis=1)
        sim_scaled = (mm.fit_transform(euclidean_distances.values.reshape(-1,1))*profile_w).reshape(1,-1)+cosine_similarities.values*candidate_w
        closest_indices = np.argsort(sim_scaled[0].tolist())[:5]
        proba= (sorted(sim_scaled[0].tolist(),reverse=True))[:5]
    return closest_indices,proba

def calculate_bmr(gender, age, height_cm, weight_kg):
    if gender == 1:
        bmr = 66 + (13.7 * weight_kg) + (5 * height_cm) - (6.8 * age)
    elif gender==0:
        bmr = 655 + (9.6 * weight_kg) + (1.7 * height_cm) - (4.7 * age)
    else:
        raise ValueError("Invalid gender. Use 'male' or 'female'.")
    return bmr

def calculate_daily_calories(bmr, activity_level):
    activity_levels = {'집에만 있음': 1.2, '약간 활동적': 1.375, '적당히 활동적': 1.55, '매우 활동적': 1.725}

    if activity_level.lower() not in activity_levels:
        raise ValueError("Invalid activity level. Use 'sedentary', 'lightly active', 'moderately active', or 'very active'.")

    total_calories = bmr * activity_levels[activity_level.lower()]
    return total_calories

def calculate_daily_nutrient_intake(Daily_caloris,upordown):
    carb_percentage = 50  # 탄수화물 비율 (%)
    protein_percentage = 20  # 단백질 비율 (%)
    fat_percentage = 30  # 지방 비율 (%)

    if upordown=='체중 증가':
        wanted_caloris = Daily_caloris*1.15
    elif upordown=='체중 유지':
        wanted_caloris = Daily_caloris*1
    else:
        wanted_caloris = Daily_caloris*0.85

    carb_calories = wanted_caloris * (carb_percentage / 100)
    protein_calories = wanted_caloris * (protein_percentage / 100)
    fat_calories = wanted_caloris * (fat_percentage / 100)

    # 1g의 탄수화물 또는 단백질은 4kcal, 1g의 지방은 9kcal이다.
    carb_grams = carb_calories / 4
    protein_grams = protein_calories / 4
    fat_grams = fat_calories / 9

    return carb_grams,protein_grams,fat_grams
