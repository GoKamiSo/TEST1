import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Tải tập dữ liệu
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Chọn các đặc trưng RM, PTRATIO và LSTAT
selected_features = data[:, [5, 10, 12]]

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(selected_features, target, test_size=0.2, random_state=42)

# Đào tạo mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

# Lưu mô hình đã đào tạo
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
