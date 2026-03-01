import pandas as pd
import joblib
import numpy as np

# 加载CatBoost模型和标准化器
catboost_model = joblib.load('catboost_model.pkl')
scaler = joblib.load('catboost_scaler.pkl')

# 加载预测数据
# 确保数据包含与训练时相同的特征列（第2-26列，共25个特征）
new_data = pd.read_csv(r'last.csv')  # 替换为你的文件路径

# 检查特征数量
if new_data.shape[1] < 25:
    raise ValueError("输入数据特征数量不足，需要25个特征列")

# 提取特征（假设格式与CatBoost训练数据相同）
if new_data.shape[1] == 25:  # 如果只有特征列
    x_new = new_data.iloc[:, :]
else:  # 如果有额外列（如序号列）
    x_new = new_data.iloc[:, 1:26]  # 使用与CatBoost训练相同的列位置（第2-26列）

# 标准化特征（使用CatBoost训练时相同的标准化器）
x_new_scaled = scaler.transform(x_new)

# 进行预测
predictions = catboost_model.predict(x_new_scaled)

# 将预测结果添加到数据框
new_data['Predicted_PCE'] = predictions

# 保存结果
new_data.to_csv('catboost_predictions_results_last.csv', index=False)
print("CatBoost预测完成！结果已保存到 catboost_predictions_results_last.csv")

# 打印前5个预测结果示例
print("\nCatBoost预测结果示例:")
print(new_data[['Predicted_PCE']].head())

# 可选：打印统计信息
print(f"\n预测统计信息:")
print(f"预测样本数: {len(predictions)}")
print(f"预测PCE范围: [{predictions.min():.3f}, {predictions.max():.3f}]")
print(f"预测PCE均值: {predictions.mean():.3f}")
print(f"预测PCE标准差: {predictions.std():.3f}")