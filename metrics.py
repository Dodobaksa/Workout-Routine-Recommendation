##############Metrics###############
def recall_at_k(actual, predicted, k=5):
    """
    actual: 실제로 발생한 아이템들의 리스트
    predicted: 모델이 예측한 상위 k개의 아이템들의 리스트
    k: 상위 몇 개의 아이템을 고려할지를 나타내는 파라미터
    """
    # 실제로 발생한 아이템들 중 상위 k개 아이템들의 수를 계산
    relevant_items = actual[:k]

    # 모델이 예측한 상위 k개 아이템들 중 실제로 발생한 아이템들의 수를 계산
    intersection = len(set(relevant_items) & set(predicted))

    # 실제로 발생한 아이템들 중에서 예측한 아이템들의 재현율 계산
    recall = intersection / min(k, len(actual))

    return recall

def ap_at_k(like_item, recommend_item, k):
    precisions = []
    # 1부터 K까지 Loop를 돌며 Precision을 계산합니다
    for i in range(k):
        base = recommend_item[:i+1]  # 리스트 슬라이싱을 통해 변경 가능성을 줄입니다.
        intersect = set(base).intersection(like_item)
        relevance = recommend_item[i] in like_item

        precisions.append(len(intersect) / len(base) * relevance)

    ap_k = sum(precisions) / len(precisions)
    return ap_k