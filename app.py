<<<<<<< HEAD
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 데이터 불러오기
df = pd.read_csv("data/diabetes.csv")

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
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step6: 예측 및 성능 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Streamlit UI 디자인
st.set_page_config(page_title="Diabetes Prediction Dashboard", layout="wide")

# 사이드바 메뉴 추가 (탭처럼 보이도록 selectbox 사용)
st.sidebar.title("Navigation")
menu = st.sidebar.selectbox("Go to", ["Home", "EDA", "Data Analysis", "Model Performance"])

# Home screen
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
    - 이 데이터셋은 당뇨병 예측을 위한 다양한 신체 지표를 포함하고 있으며, 각 샘플은 **당뇨병**의 발병 여부를 예측하는 데 중요한 역할을 합니다.
    
    #### 당뇨병 데이터 개요
    - **Pregnancies**: 당뇨병에 영향을 줄 수 있는 임신 횟수. 임신과 관련된 호르몬 변화가 인슐린 저항성에 영향을 미칠 수 있습니다.
    - **Glucose**: 혈액 내 포도당 수치. 당뇨병의 주요한 지표인 포도당 수치는 혈당 조절 능력에 직접적인 영향을 미칩니다.
    - **BloodPressure**: 혈압 수치. 고혈압은 당뇨병과 밀접한 연관이 있습니다.
    - **SkinThickness**: 피부 두께. 인슐린 저항성과 관련된 신체의 변화 중 하나로, 피부 두께가 두꺼운 경우 당뇨병 위험이 증가할 수 있습니다.
    - **Insulin**: 인슐린 수치. 인슐린은 혈당을 조절하는 호르몬으로, 당뇨병과 밀접한 연관이 있습니다.
    - **BMI**: 체질량지수. 비만은 당뇨병과 관련이 있으며, 높은 BMI는 당뇨병 발병 위험을 증가시킬 수 있습니다.
    - **DiabetesPedigreeFunction**: 가족력 지수. 가족 중 당뇨병 환자가 있을 경우 당뇨병 발병 확률이 높아지므로 중요한 지표입니다.
    - **Age**: 나이. 나이가 많을수록 당뇨병 발병 위험이 커지는 경향이 있습니다.
    - **Outcome**: 당뇨 여부 (0: 정상, 1: 당뇨병). 이 타겟 변수는 예측 모델이 예측하려는 값으로, 당뇨병의 유무를 나타냅니다.
    """)

# Data Analysis screen
def data_analysis():
    st.title("Data Analysis")
    
    # 특성 설명
    st.subheader("### 특성 설명")
    st.markdown(""" 
    - **Pregnancies**: 임신 횟수
    - **Glucose**: 포도당 수치
    - **BloodPressure**: 혈압
    - **SkinThickness**: 피부 두께
    - **Insulin**: 인슐린 수치
    - **BMI**: 체질량지수
    - **DiabetesPedigreeFunction**: 가족력 지수
    - **Age**: 나이
    - **Outcome**: 당뇨 여부 (0: 정상, 1: 당뇨병)
    """)
    
    # 상위 데이터 표시
    with st.expander("상위 5개 데이터"):
        st.write(df.head())

    # 데이터 통계 표시
    with st.expander("데이터 통계"):
        st.write(df.describe())

    # 컬럼 데이터 표시
    with st.expander("컬럼 데이터"):
        st.write(df.columns)

    # 조건 데이터를 표시 (예: BMI가 30 이상인 데이터)
    with st.expander("BMI가 30 이상인 데이터"):
        st.write(df[df['BMI'] >= 30])

# Exploratory Data Analysis (EDA)
def eda():
    st.title("Exploratory Data Analysis")
    chart_tabs = st.selectbox("Choose a chart", ["Histogram", "Boxplot", "Heatmap"])

    if chart_tabs == "Histogram":
        st.subheader("Feature Distribution")
        # Plotly Histograms for feature distribution
        columns = ["Glucose", "BloodPressure", "BMI", "Age"]
        for col in columns:
            fig = px.histogram(df, x=col, nbins=20, title=f"{col} Distribution", marginal="box")
            fig.update_layout(width=600, height=400)  # Adjust size here
            st.plotly_chart(fig)

    elif chart_tabs == "Boxplot":
        st.subheader("Glucose Boxplot by Outcome")
        # Plotly Boxplot for Glucose by Outcome
        fig = px.box(df, x="Outcome", y="Glucose", title="Glucose Boxplot by Outcome")
        fig.update_layout(width=600, height=400)  # Adjust size here
        st.plotly_chart(fig)

        st.subheader("BMI Boxplot by Outcome")
        # Plotly Boxplot for BMI by Outcome
        fig = px.box(df, x="Outcome", y="BMI", title="BMI Boxplot by Outcome")
        fig.update_layout(width=600, height=400)  # Adjust size here
        st.plotly_chart(fig)

    elif chart_tabs == "Heatmap":
        st.subheader("Feature Correlation Heatmap")
        # Plotly Heatmap for feature correlation
        corr = df.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='Viridis',
            zmin=-1, zmax=1
        ))
        fig.update_layout(title="Correlation Heatmap", xaxis_title="Features", yaxis_title="Features",
                          width=600, height=400)  # Adjust size here
        st.plotly_chart(fig)

# Model Performance
def model_performance():
    st.title("Model Performance")
    st.write(f'### Model Accuracy: {accuracy:.2f}')
    
    # Convert classification report to a pandas DataFrame
    classification_rep_dict = classification_report(y_test, y_pred, output_dict=True)
    classification_rep_df = pd.DataFrame(classification_rep_dict).transpose()

    # Display the classification report as a table
    st.subheader("Classification Report")
    st.dataframe(classification_rep_df)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=["Normal", "Diabetes"],
        y=["Normal", "Diabetes"],
        colorscale='Blues',
        zmin=0, zmax=np.max(conf_matrix),
        showscale=True
    ))
    fig.update_layout(
        title="Confusion Matrix", 
        xaxis_title="Predicted", 
        yaxis_title="Actual", 
        width=600, 
        height=400,
        xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=["Normal", "Diabetes"]),
        yaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=["Normal", "Diabetes"])
    )  # Adjust size and axis ticks for clarity
    st.plotly_chart(fig)

# 메뉴 선택에 따른 화면 전환
if menu == "Home":
    home()
elif menu == "EDA":
    eda()
elif menu == "Data Analysis":
    data_analysis()
elif menu == "Model Performance":
    model_performance()
=======
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 데이터 불러오기
df = pd.read_csv("data/diabetes.csv")

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
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step6: 예측 및 성능 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Streamlit UI 디자인
st.set_page_config(page_title="Diabetes Prediction Dashboard", layout="wide")

# 사이드바 메뉴 추가 (탭처럼 보이도록 selectbox 사용)
st.sidebar.title("Navigation")
menu = st.sidebar.selectbox("Go to", ["Home", "EDA", "Data Analysis", "Model Performance"])

# Home screen
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
    - 이 데이터셋은 당뇨병 예측을 위한 다양한 신체 지표를 포함하고 있으며, 각 샘플은 **당뇨병**의 발병 여부를 예측하는 데 중요한 역할을 합니다.
    
    #### 당뇨병 데이터 개요
    - **Pregnancies**: 당뇨병에 영향을 줄 수 있는 임신 횟수. 임신과 관련된 호르몬 변화가 인슐린 저항성에 영향을 미칠 수 있습니다.
    - **Glucose**: 혈액 내 포도당 수치. 당뇨병의 주요한 지표인 포도당 수치는 혈당 조절 능력에 직접적인 영향을 미칩니다.
    - **BloodPressure**: 혈압 수치. 고혈압은 당뇨병과 밀접한 연관이 있습니다.
    - **SkinThickness**: 피부 두께. 인슐린 저항성과 관련된 신체의 변화 중 하나로, 피부 두께가 두꺼운 경우 당뇨병 위험이 증가할 수 있습니다.
    - **Insulin**: 인슐린 수치. 인슐린은 혈당을 조절하는 호르몬으로, 당뇨병과 밀접한 연관이 있습니다.
    - **BMI**: 체질량지수. 비만은 당뇨병과 관련이 있으며, 높은 BMI는 당뇨병 발병 위험을 증가시킬 수 있습니다.
    - **DiabetesPedigreeFunction**: 가족력 지수. 가족 중 당뇨병 환자가 있을 경우 당뇨병 발병 확률이 높아지므로 중요한 지표입니다.
    - **Age**: 나이. 나이가 많을수록 당뇨병 발병 위험이 커지는 경향이 있습니다.
    - **Outcome**: 당뇨 여부 (0: 정상, 1: 당뇨병). 이 타겟 변수는 예측 모델이 예측하려는 값으로, 당뇨병의 유무를 나타냅니다.
    """)

# Data Analysis screen
def data_analysis():
    st.title("Data Analysis")
    
    # 특성 설명
    st.subheader("### 특성 설명")
    st.markdown(""" 
    - **Pregnancies**: 임신 횟수
    - **Glucose**: 포도당 수치
    - **BloodPressure**: 혈압
    - **SkinThickness**: 피부 두께
    - **Insulin**: 인슐린 수치
    - **BMI**: 체질량지수
    - **DiabetesPedigreeFunction**: 가족력 지수
    - **Age**: 나이
    - **Outcome**: 당뇨 여부 (0: 정상, 1: 당뇨병)
    """)
    
    # 상위 데이터 표시
    with st.expander("상위 5개 데이터"):
        st.write(df.head())

    # 데이터 통계 표시
    with st.expander("데이터 통계"):
        st.write(df.describe())

    # 컬럼 데이터 표시
    with st.expander("컬럼 데이터"):
        st.write(df.columns)

    # 조건 데이터를 표시 (예: BMI가 30 이상인 데이터)
    with st.expander("BMI가 30 이상인 데이터"):
        st.write(df[df['BMI'] >= 30])

# Exploratory Data Analysis (EDA)
def eda():
    st.title("Exploratory Data Analysis")
    chart_tabs = st.selectbox("Choose a chart", ["Histogram", "Boxplot", "Heatmap"])

    if chart_tabs == "Histogram":
        st.subheader("Feature Distribution")
        # Plotly Histograms for feature distribution
        columns = ["Glucose", "BloodPressure", "BMI", "Age"]
        for col in columns:
            fig = px.histogram(df, x=col, nbins=20, title=f"{col} Distribution", marginal="box")
            fig.update_layout(width=600, height=400)  # Adjust size here
            st.plotly_chart(fig)

    elif chart_tabs == "Boxplot":
        st.subheader("Glucose Boxplot by Outcome")
        # Plotly Boxplot for Glucose by Outcome
        fig = px.box(df, x="Outcome", y="Glucose", title="Glucose Boxplot by Outcome")
        fig.update_layout(width=600, height=400)  # Adjust size here
        st.plotly_chart(fig)

        st.subheader("BMI Boxplot by Outcome")
        # Plotly Boxplot for BMI by Outcome
        fig = px.box(df, x="Outcome", y="BMI", title="BMI Boxplot by Outcome")
        fig.update_layout(width=600, height=400)  # Adjust size here
        st.plotly_chart(fig)

    elif chart_tabs == "Heatmap":
        st.subheader("Feature Correlation Heatmap")
        # Plotly Heatmap for feature correlation
        corr = df.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='Viridis',
            zmin=-1, zmax=1
        ))
        fig.update_layout(title="Correlation Heatmap", xaxis_title="Features", yaxis_title="Features",
                          width=600, height=400)  # Adjust size here
        st.plotly_chart(fig)

# Model Performance
def model_performance():
    st.title("Model Performance")
    st.write(f'### Model Accuracy: {accuracy:.2f}')
    
    # Convert classification report to a pandas DataFrame
    classification_rep_dict = classification_report(y_test, y_pred, output_dict=True)
    classification_rep_df = pd.DataFrame(classification_rep_dict).transpose()

    # Display the classification report as a table
    st.subheader("Classification Report")
    st.dataframe(classification_rep_df)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=["Normal", "Diabetes"],
        y=["Normal", "Diabetes"],
        colorscale='Blues',
        zmin=0, zmax=np.max(conf_matrix),
        showscale=True
    ))
    fig.update_layout(
        title="Confusion Matrix", 
        xaxis_title="Predicted", 
        yaxis_title="Actual", 
        width=600, 
        height=400,
        xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=["Normal", "Diabetes"]),
        yaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=["Normal", "Diabetes"])
    )  # Adjust size and axis ticks for clarity
    st.plotly_chart(fig)

# 메뉴 선택에 따른 화면 전환
if menu == "Home":
    home()
elif menu == "EDA":
    eda()
elif menu == "Data Analysis":
    data_analysis()
elif menu == "Model Performance":
    model_performance()