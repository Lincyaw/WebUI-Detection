def calculate_f1(precision, recall):
    """计算 F1 分数"""
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# 表中的数据
data = [
    {"Precision": 53.3, "Recall": 56.5},
    {"Precision": 56.9, "Recall": 59.7},
    {"Precision": 53.7, "Recall": 56.6},
    {"Precision": 56.8, "Recall": 60.5},
    {"Precision": 58.3, "Recall": 62.2}
]

# 计算 F1 分数
for row in data:
    precision = row["Precision"] / 100  # 将百分比转换为小数
    recall = row["Recall"] / 100        # 将百分比转换为小数
    f1_score = calculate_f1(precision, recall)
    print(f"F1 Score: {f1_score:.4f}")  # 打印结果，保留四位小数
