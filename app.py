import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# **✅ Streamlit 페이지 설정 (가장 위에 위치해야 함)**
st.set_page_config(page_title="Diabetes Prediction Dashboard", layout="wide")

# **데이터 불러오기**
@st.cache_data
def load_data():
    return pd.read_csv("data/diabetes.csv")

df = load_data()

# Step1: 결측치 확인 및 이상치 제거
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

df = remove_outliers(df, 'BMI')

# Step2: 특성과 타겟 분리
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Step3: 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step4: 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step5: 모델 학습
@st.cache_resource
def train_model():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

# Step6: 예측 및 성능 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# **✅ Streamlit UI 시작**

# 사이드바 메뉴 추가
st.sidebar.title("Navigation")
menu = st.sidebar.selectbox("Go to", ["Home", "EDA", "Data Analysis", "Model Performance"])

# **Home 화면**
def home():
    st.title("Diabetes Prediction Dashboard")
    st.markdown("""
    ### 데이터 출처
    이 프로젝트에 사용된 데이터는 Kaggle의 **Pima Indians Diabetes Database**에서 가져왔습니다.
    [Pima Indians Diabetes Dataset on Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
    
    ### 데이터 설명
    이 데이터셋은 **Pima Indians**의 여성들에게서 수집된 건강 데이터를 포함하고 있습니다. 주로 **당뇨병**의 유무를 예측하는 데 사용됩니다.
    - **diabetes.csv** 파일은 8개의 특성(feature)과 1개의 타겟 변수(target)인 **Outcome**을 포함하고 있습니다.
    - **Outcome**이 0인 경우 **당뇨병이 없다**고 표시되며, 1인 경우 **당뇨병이 있다**고 표시됩니다.
    """)

# **EDA (탐색적 데이터 분석)**
def eda():
    st.title("Exploratory Data Analysis")
    chart_tabs = st.selectbox("Choose a chart", ["Histogram", "Boxplot", "Heatmap"])
    
    if chart_tabs == "Histogram":
        for col in ["Glucose", "BloodPressure", "BMI", "Age"]:
            fig = px.histogram(df, x=col, nbins=20, title=f"{col} Distribution", marginal="box")
            st.plotly_chart(fig)

    elif chart_tabs == "Boxplot":
        for col in ["Glucose", "BMI"]:
            fig = px.box(df, x="Outcome", y=col, title=f"{col} Boxplot by Outcome")
            st.plotly_chart(fig)

    elif chart_tabs == "Heatmap":
        corr = df.corr()
        fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='Viridis'))
        fig.update_layout(title="Feature Correlation Heatmap")
        st.plotly_chart(fig)

# **데이터 분석 화면**
def data_analysis():
    st.title("Data Analysis")
    with st.expander("상위 5개 데이터"):
        st.write(df.head())
    with st.expander("데이터 통계"):
        st.write(df.describe())
    with st.expander("컬럼 데이터"):
        st.write(df.columns)

# **모델 성능 평가 화면**
def model_performance():
    st.title("Model Performance")
    st.write(f'### Model Accuracy: {accuracy:.2f}')
    
    classification_rep_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    st.subheader("Classification Report")
    st.dataframe(classification_rep_df)
    
    st.subheader("Confusion Matrix")
    fig = go.Figure(data=go.Heatmap(z=conf_matrix, x=["Normal", "Diabetes"], y=["Normal", "Diabetes"], colorscale='Blues'))
    st.plotly_chart(fig)

# **메뉴 선택에 따른 화면 전환**
if menu == "Home":
    home()
elif menu == "EDA":
    eda()
elif menu == "Data Analysis":
    data_analysis()
elif menu == "Model Performance":
    model_performance()
