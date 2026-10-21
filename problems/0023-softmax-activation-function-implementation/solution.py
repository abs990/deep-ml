import math

def softmax(scores: list[float]) -> list[float]:
    max_score = max(scores)
    scores = [x - max_score for x in scores]
    scores = [math.exp(x) for x in scores]
    exp_sum = sum(scores)
    return [x/exp_sum for x in scores]