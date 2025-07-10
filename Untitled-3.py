# %%
import numpy as np
import pandas as pd
import zipfile
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
from pathlib import Path

# 1. 加载数据
try:
    zip_path = Path(r'C:\Users\liuziyu\Downloads\household_power_consumption.zip')
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        # 查找文本文件
        txt_files = [f for f in z.namelist() if f.endswith('.txt')]
        if not txt_files:
            raise FileNotFoundError("ZIP中没有找到.txt文件")
            
        with z.open(txt_files[0]) as f:
            df = pd.read_csv(f, sep=';', low_memory=False, na_values=['?'])
    
    # 2. 处理日期时间
    df['datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        format='%d/%m/%Y %H:%M:%S',
        dayfirst=True
    )
    df.drop(['Date', 'Time'], axis=1, inplace=True)
    df.dropna(inplace=True)
    
    # 3. 打印日期范围
    print("Start Date:", df['datetime'].min())
    print("End Date:", df['datetime'].max())
    
    # 4. 划分数据集
    train = df[df['datetime'] <= '2009-12-31']
    test = df[df['datetime'] > '2009-12-31']
    
except Exception as e:
    print(f"发生错误: {str(e)}")

# %%
# 从ZIP加载数
import zipfile 
with zipfile.ZipFile('C:\\Users\\liuziyu\\Downloads\\household_power_consumption.zip', 'r') as z:
    with z.open('household_power_consumption.txt') as f:
        df = pd.read_csv(f, sep=';', low_memory=False, na_values=['?'])

# %%
# 数据预处理
def preprocess(df):
    # 合并日期时间列
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.drop(['Date', 'Time'], axis=1, inplace=True)
    
    # 转换为数值类型
    for col in df.columns[:-1]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 处理缺失值
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    return df

df = preprocess(df)
print(f"Start Date: {df['datetime'].min()}\nEnd Date: {df['datetime'].max()}")

# %%
# 划分数据集
train = df[df['datetime'] <= '2009-12-31']
test = df[df['datetime'] > '2009-12-31']

# %%
# 数据标准化
from sklearn.preprocessing import MinMaxScaler  # 添加这行导入
# 按照时间划分训练集和测试集（假设df已包含datetime列）
train = df[df['datetime'] <= '2009-12-31']
test = df[df['datetime'] > '2009-12-31']

# 选择需要标准化的列（排除datetime列）
cols_to_scale = [col for col in df.columns if col != 'datetime']

scaler = MinMaxScaler()
cols_to_scale = [col for col in df.columns if col != 'datetime']
train_scaled = scaler.fit_transform(train[cols_to_scale])
test_scaled = scaler.transform(test[cols_to_scale])

# %%
# 创建时间序列数据集
def create_sequences(data, seq_length=24, target_col=0):
    xs, ys = [], []
    for i in range(len(data)-seq_length-1):
        xs.append(data[i:(i+seq_length)])
        ys.append(data[i+seq_length, target_col])  # 预测Global_active_power
    return np.array(xs), np.array(ys)

X_train, y_train = create_sequences(train_scaled)
X_test, y_test = create_sequences(test_scaled)

# %%
# 创建PyTorch数据集
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# %%
# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(
    input_size=X_train.shape[2],
    hidden_size=50,
    num_layers=2,
    output_size=1
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 20
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# %%
# 测试集评估
model.eval()
test_preds, test_true = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        test_preds.extend(outputs.cpu().numpy())
        test_true.extend(labels.cpu().numpy())

# 反归一化
dummy = np.zeros((len(test_preds), len(cols_to_scale)))
dummy[:, 0] = test_preds
test_preds = scaler.inverse_transform(dummy)[:, 0]

dummy[:, 0] = test_true
test_true = scaler.inverse_transform(dummy)[:, 0]

# 计算RMSE
rmse = np.sqrt(np.mean((test_true - test_preds)**2))
print(f'Test RMSE: {rmse:.2f}')

# %%
# 绘制结果
plt.figure(figsize=(12, 6))
plt.plot(test_true[:200], label='True')
plt.plot(test_preds[:200], label='Predicted')
plt.title('Household Power Consumption Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Global Active Power')
plt.legend()
plt.show()