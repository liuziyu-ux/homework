# -*- coding: utf-8 -*-
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. 解决文件路径问题 ==============================================
# 定义可能的文件路径（根据你的实际情况修改）
possible_paths = [
    'data/train.csv',  # 相对路径
    r'C:\Users\liuziyu\Downloads\train.csv',  # 绝对路径1
    r'C:\Users\liuziyu\data\train.csv'  # 绝对路径2
]

# 查找真实存在的文件路径
file_path = None
for path in possible_paths:
    if os.path.exists(path):
        file_path = path
        break

if file_path is None:
    raise FileNotFoundError(
        "未找到train.csv文件，请检查：\n"
        "1. 文件是否存在于以下位置之一：\n"
        f"{possible_paths}\n"
        "2. 文件名是否正确（注意大小写）\n"
        "3. 如果使用相对路径，当前工作目录是否正确"
    )

# 2. 主程序 ======================================================
try:
    # Load data
    data = pd.read_csv(file_path)  # 使用找到的正确路径
    df = data.copy()

    # Data preprocessing
    df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
    df.dropna(inplace=True)
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'])

    # Separate features and labels
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build and evaluate models
    models = {
        "SVM": SVC(kernel='rbf', random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

    # Display results
    for model_name, metrics in results.items():
        print(f"\n{model_name} Performance:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print("Confusion Matrix:")
        print(metrics['confusion_matrix'])

except Exception as e:
    print(f"程序出错: {str(e)}")
    print("建议检查：")
    print("1. 所有必需的库是否已安装（pandas, scikit-learn）")
    print("2. 数据文件内容是否完整")