import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import ShuffleSplit, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.ticker as mticker
import warnings
from catboost import CatBoostRegressor
import statsmodels.api as sm
import joblib
from matplotlib import rcParams

# 设置临时文件夹为英文路径（首选）
import os, tempfile
os.environ['TMP'] = r'C:\Temp'  # 需手动创建此文件夹
os.environ['TEMP'] = r'C:\Temp'
tempfile.tempdir = r'C:\Temp'

# 可视化配置 - 增加字体大小并加粗
plt.rcParams.update({
    'font.size': 8,  # 减小字体大小以适应小画布
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'lines.linewidth': 1.0,  # 减小线宽
    'font.family': 'Times New Roman',
    'font.weight': 'bold',  # 全局字体加粗
    'axes.titleweight': 'bold',  # 标题加粗
    'axes.labelweight': 'bold',  # 坐标轴标签加粗
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none'
})

# 导入文件
data = pd.read_csv('Molecular descriptors - dataset.csv', header=0)

# 特征与目标变量分离
x = data.iloc[:, 1:26]
y = data.iloc[:, 26]

# 交叉验证设置
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

# 划分训练集和测试集
x_train0, x_test0, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# 标准化处理
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train0)
x_test = scaler.transform(x_test0)

# catBoost
warnings.filterwarnings('ignore')

param_test1 = {
    'max_depth': [2, 4, 6],
    'learning_rate': [0.1, 0.05, 0.001],
    'l2_leaf_reg': [1e-3, 1e-2],
    'iterations': [50, 30, 10],
}

cat = CatBoostRegressor(random_state=42, verbose=0)
gsearch = GridSearchCV(cat, param_grid=param_test1, scoring='neg_mean_squared_error', cv=10, n_jobs=8)
gsearch.fit(x_train, y_train)
print(f"Best parameters: {gsearch.best_params_}")
cat = gsearch.best_estimator_

train_pred = cat.predict(x_train)
test_pred = cat.predict(x_test)

print(f"RMSE_train: {np.sqrt(mean_squared_error(y_train, train_pred))}")
print(f"R_train: {stats.pearsonr(y_train, train_pred)[0]}")
print(f"RMSE_test: {np.sqrt(mean_squared_error(y_test, test_pred))}")
print(f"R_test: {stats.pearsonr(y_test, test_pred)[0]}")
print(f"R²_train: {r2_score(y_train, train_pred)}")
print(f"R²_test: {r2_score(y_test, test_pred)}")

# 计算各项指标
train_mae = mean_absolute_error(y_train, train_pred)
train_mape = mean_absolute_percentage_error(y_train, train_pred)
train_mse = mean_squared_error(y_train, train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, train_pred)

test_mae = mean_absolute_error(y_test, test_pred)
test_mape = mean_absolute_percentage_error(y_test, test_pred)
test_mse = mean_squared_error(y_test, test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, test_pred)

metrics = {
    'Train': {
        'MAE': train_mae,
        'MAPE': train_mape,
        'MSE': train_mse,
        'RMSE': train_rmse,
        'R2': train_r2,
    },
    'Test': {
        'MAE': test_mae,
        'MAPE': test_mape,
        'MSE': test_mse,
        'RMSE': test_rmse,
        'R2': test_r2,
    },
}

metrics_path = 'catboost_performance_metrics.txt'
with open(metrics_path, 'w', encoding='utf-8') as f:
    for split, values in metrics.items():
        f.write(f'{split} Metrics:\n')
        for metric_name, metric_value in values.items():
            f.write(f'  {metric_name}: {metric_value:.6f}\n')
        f.write('\n')

print(f'Metrics saved to {metrics_path}')

def compute_regression_ci(actual_values: np.ndarray, predicted_values: np.ndarray):
    design_matrix = sm.add_constant(actual_values)
    regression_model = sm.OLS(predicted_values, design_matrix).fit()
    x_seq = np.linspace(actual_values.min(), actual_values.max(), 200)
    x_seq_design = sm.add_constant(x_seq)
    prediction_res = regression_model.get_prediction(x_seq_design)
    summary = prediction_res.summary_frame(alpha=0.05)
    return (
        x_seq,
        summary['mean'].to_numpy(),
        summary['mean_ci_lower'].to_numpy(),
        summary['mean_ci_upper'].to_numpy(),
    )

train_actual = y_train.to_numpy()
test_actual = y_test.to_numpy()

(train_x_seq, train_line_mean, train_ci_lower, train_ci_upper) = compute_regression_ci(
    train_actual,
    train_pred,
)
(test_x_seq, test_line_mean, test_ci_lower, test_ci_upper) = compute_regression_ci(
    test_actual,
    test_pred,
)

# 将厘米转换为英寸 (1英寸=2.54厘米)
width_cm = 8
height_cm = 6
width_inch = width_cm / 2.54
height_inch = height_cm / 2.54

# 创建图形，设置尺寸为8cm×6cm
plt.figure(dpi=900, figsize=(width_inch, height_inch))
train_color = '#4FBDFF'
test_color = '#FB6F6F'

# 绘制散点图
plt.scatter(
    train_actual,
    train_pred,
    color=train_color,
    alpha=0.7,
    s=8,
    marker='P',
    label='Train',
)
plt.scatter(
    test_actual,
    test_pred,
    color=test_color,
    alpha=0.7,
    s=8,
    marker='^',
    label='Test',
)

# 绘制回归线和置信区间
plt.plot(
    train_x_seq,
    train_line_mean,
    color=train_color,
    linewidth=1.0,
    label='Train regerss fit',
)
plt.fill_between(
    train_x_seq,
    train_ci_lower,
    train_ci_upper,
    color=train_color,
    alpha=0.15,
    label='Train 95% conf.bounds',
)

plt.plot(
    test_x_seq,
    test_line_mean,
    color=test_color,
    linewidth=1.0,
    label='Test regerss fit',
)
plt.fill_between(
    test_x_seq,
    test_ci_lower,
    test_ci_upper,
    color=test_color,
    alpha=0.15,
    label='Test 95% conf.bounds',
)

# 添加R²文本
plt.text(
    11,
    26,
    f'Train: R² = {train_r2:.3f}',
    color='black',
    fontsize=5.5,
)
plt.text(
    11,
    25,
    f'Test: R² = {test_r2:.3f}',
    color='black',
    fontsize=5.5,
)

# 设置坐标轴范围和比例
plt.gca().set_aspect('equal')
#plt.title('CatBoost Performance')
plt.minorticks_on()

# 设置坐标轴刻度
ax = plt.gca()

# 设置x轴和y轴的主刻度为2，次刻度为1
from matplotlib.ticker import MultipleLocator
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(2))
ax.yaxis.set_minor_locator(MultipleLocator(1))

# 设置刻度标签格式
plt.xlabel('Measured PCE(%)', fontsize=9, fontweight='bold', labelpad=5)
plt.ylabel('Predicted PCE(%)', fontsize=9, fontweight='bold', labelpad=5)

# 设置坐标轴范围
plt.xlim(10, 28)
plt.ylim(10, 28)  # 注意：这里y轴范围改为10-26，与示例保持一致

# 添加网格线（只显示主网格线）
plt.grid(True, which='major', linestyle='--', linewidth=0.3, alpha=0.7)
plt.grid(False, which='minor')

# 添加图例
plt.legend(loc='lower right', fontsize=5.5, frameon=True, fancybox=True, shadow=True,
           markerscale=0.7, handlelength=1.2, borderpad=0.3, labelspacing=0.2)

# 调整布局，确保所有元素都适应小画布
plt.tight_layout(pad=1.0)

# 保存图形
plt.savefig(
    'CatBoost_Performance.png',
    dpi=900,
    bbox_inches='tight',
    pad_inches=0.05,
    facecolor='white',
)

plt.show()

# 保存最佳CatBoost模型
joblib.dump(cat, 'catboost_model.pkl')
print(f"CatBoost模型已保存为 'catboost_model.pkl'")

# 保存标准化器
joblib.dump(scaler, 'catboost_scaler.pkl')
print(f"标准化器已保存为 'catboost_scaler.pkl'")