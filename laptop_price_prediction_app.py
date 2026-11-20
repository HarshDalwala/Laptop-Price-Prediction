import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Laptop Price Prediction App",
    page_icon="💻",
    layout="wide"
)

st.title("💻 Laptop Price Prediction App")
st.caption("Explore laptop data, train regression models, and predict laptop prices.")


# ---------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------
BASE_DIR = Path("C:\\Users\\admin\\Desktop\\Tops Technology\\Machine Learning\\data set\\0 Projects\\1 Regression\\1 laptop price\\Laptop_Price_Prediction.csv").resolve().parent
DATASET_PATH = BASE_DIR / "Laptop_Price_Prediction.csv"

REQUIRED_COLUMNS = [
    "Brand",
    "Processor_Speed",
    "RAM_Size",
    "Storage_Capacity",
    "Screen_Size",
    "Weight",
    "Price"
]

FEATURE_COLUMNS = [
    "Brand",
    "Processor_Speed",
    "RAM_Size",
    "Storage_Capacity",
    "Screen_Size",
    "Weight"
]
TARGET_COLUMN = "Price"
NUMERIC_FEATURES = [
    "Processor_Speed",
    "RAM_Size",
    "Storage_Capacity",
    "Screen_Size",
    "Weight"
]
CATEGORICAL_FEATURES = ["Brand"]


if not DATASET_PATH.exists():
    st.error(
        "Dataset file not found. Please keep "
        "'Laptop_Price_Prediction.csv' in the same folder as this Python file."
    )
    st.stop()


@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)


df = load_data(DATASET_PATH)

missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
if missing_columns:
    st.error(f"Required columns are missing from the dataset: {missing_columns}")
    st.stop()

if df.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()

# Keep only the columns required for this project.
df = df[REQUIRED_COLUMNS].copy()


# ---------------------------------------------------------
# Train-Test Split
# ---------------------------------------------------------
def split_data(test_size=0.2, random_state=42):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------
def create_one_hot_encoder():
    # Supports both newer and older scikit-learn versions.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def create_preprocessor():
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", create_one_hot_encoder())
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )


# ---------------------------------------------------------
# Model Creation
# ---------------------------------------------------------
MODEL_NAMES = [
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
    "Linear Regression",
    "Ridge Regression",
    "Lasso Regression",
    "K-Nearest Neighbors",
    "AdaBoost",
    "Support Vector Regression",
    "Polynomial Regression"
]


def create_model(model_name, degree=2):
    if model_name == "Decision Tree":
        estimator = DecisionTreeRegressor(random_state=42)

    elif model_name == "Random Forest":
        estimator = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

    elif model_name == "Gradient Boosting":
        estimator = GradientBoostingRegressor(random_state=42)

    elif model_name == "Linear Regression":
        estimator = LinearRegression()

    elif model_name == "Ridge Regression":
        estimator = Ridge()

    elif model_name == "Lasso Regression":
        estimator = Lasso(max_iter=10000)

    elif model_name == "K-Nearest Neighbors":
        estimator = KNeighborsRegressor(n_neighbors=5)

    elif model_name == "AdaBoost":
        estimator = AdaBoostRegressor(random_state=42)

    elif model_name == "Support Vector Regression":
        estimator = SVR()

    elif model_name == "Polynomial Regression":
        return Pipeline(steps=[
            ("preprocessor", create_preprocessor()),
            ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
            ("model", LinearRegression())
        ])

    else:
        estimator = LinearRegression()

    return Pipeline(steps=[
        ("preprocessor", create_preprocessor()),
        ("model", estimator)
    ])


# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
if "current_page" not in st.session_state:
    st.session_state.current_page = "Show Data Summary"

st.sidebar.title("📌 Navigation")

pages = [
    "Show Data Summary",
    "EDA - Exploratory Data Analysis",
    "Show Train-Test Split Summary",
    "Show Model Selection",
    "Prediction"
]

for page in pages:
    if st.sidebar.button(page, use_container_width=True):
        st.session_state.current_page = page

st.sidebar.markdown("---")
st.sidebar.caption("Laptop Price Prediction | Regression")


# ---------------------------------------------------------
# 1. Data Summary
# ---------------------------------------------------------
if st.session_state.current_page == "Show Data Summary":
    st.header("📊 Data Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Dataset Information")
    info_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": [str(dtype) for dtype in df.dtypes],
        "Missing Values": [int(value) for value in df.isnull().sum()],
        "Unique Values": [int(df[col].nunique()) for col in df.columns]
    })
    st.dataframe(info_df, use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)


# ---------------------------------------------------------
# 2. EDA - Exploratory Data Analysis
# ---------------------------------------------------------
elif st.session_state.current_page == "EDA - Exploratory Data Analysis":
    st.header("📈 Exploratory Data Analysis")

    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"]
    )

    if analysis_type == "Univariate Analysis":
        st.subheader("Univariate Analysis")

        selected_column = st.selectbox(
            "Select Column",
            df.columns.tolist()
        )

        if df[selected_column].dtype == "object":
            count_df = df[selected_column].value_counts().reset_index()
            count_df.columns = [selected_column, "Count"]

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    px.bar(count_df, x=selected_column, y="Count",
                           title=f"{selected_column} Count"),
                    use_container_width=True
                )
            with col2:
                st.plotly_chart(
                    px.pie(count_df, names=selected_column, values="Count",
                           title=f"{selected_column} Distribution"),
                    use_container_width=True
                )
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    px.histogram(df, x=selected_column, title=f"{selected_column} Histogram"),
                    use_container_width=True
                )
            with col2:
                st.plotly_chart(
                    px.box(df, y=selected_column, title=f"{selected_column} Box Plot"),
                    use_container_width=True
                )

            st.plotly_chart(
                px.violin(df, y=selected_column, box=True,
                          title=f"{selected_column} Violin Plot"),
                use_container_width=True
            )

    elif analysis_type == "Bivariate Analysis":
        st.subheader("Bivariate Analysis")

        col1, col2 = st.columns(2)
        with col1:
            x_column = st.selectbox(
                "Select X-Axis Column",
                df.columns.tolist(),
                index=1
            )
        with col2:
            y_column = st.selectbox(
                "Select Y-Axis Column",
                df.columns.tolist(),
                index=6
            )

        if x_column == y_column:
            st.warning("Please select two different columns.")
        else:
            if df[x_column].dtype == "object":
                st.plotly_chart(
                    px.box(df, x=x_column, y=y_column,
                           title=f"{y_column} by {x_column}"),
                    use_container_width=True
                )
                st.plotly_chart(
                    px.bar(
                        df.groupby(x_column, as_index=False)[y_column].mean(),
                        x=x_column, y=y_column,
                        title=f"Average {y_column} by {x_column}"
                    ),
                    use_container_width=True
                )
            else:
                st.plotly_chart(
                    px.scatter(
                        df, x=x_column, y=y_column,
                        color="Brand",
                        trendline="ols",
                        title=f"{x_column} vs {y_column}"
                    ),
                    use_container_width=True
                )
                st.plotly_chart(
                    px.density_heatmap(
                        df, x=x_column, y=y_column,
                        title=f"{x_column} and {y_column} Density"
                    ),
                    use_container_width=True
                )

    else:
        st.subheader("Multivariate Analysis")

        numeric_df = df.select_dtypes(include=np.number)

        st.plotly_chart(
            px.imshow(
                numeric_df.corr(),
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Correlation Heatmap"
            ),
            use_container_width=True
        )

        st.plotly_chart(
            px.scatter_matrix(
                df,
                dimensions=NUMERIC_FEATURES + [TARGET_COLUMN],
                color="Brand",
                title="Scatter Matrix"
            ),
            use_container_width=True
        )


# ---------------------------------------------------------
# 3. Train-Test Split Summary
# ---------------------------------------------------------
elif st.session_state.current_page == "Show Train-Test Split Summary":
    st.header("✂️ Train-Test Split Summary")

    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider(
            "Select Test Size",
            min_value=0.10,
            max_value=0.40,
            value=0.20,
            step=0.05
        )
    with col2:
        random_state = st.number_input(
            "Enter Random State",
            min_value=0,
            max_value=1000,
            value=42,
            step=1
        )

    X_train, X_test, y_train, y_test = split_data(
        test_size=test_size,
        random_state=int(random_state)
    )

    st.subheader("Split Dataset Shapes")

    col1, col2 = st.columns(2)
    col1.metric("Training Records", X_train.shape[0])
    col2.metric("Testing Records", X_test.shape[0])

    shape_df = pd.DataFrame({
        "Dataset": ["X_train", "X_test", "y_train", "y_test"],
        "Shape": [
            str(X_train.shape),
            str(X_test.shape),
            str(y_train.shape),
            str(y_test.shape)
        ]
    })
    st.dataframe(shape_df, use_container_width=True)

    st.subheader("Training Data Preview")
    st.dataframe(X_train.head(), use_container_width=True)

    st.subheader("Testing Data Preview")
    st.dataframe(X_test.head(), use_container_width=True)


# ---------------------------------------------------------
# 4. Model Selection
# ---------------------------------------------------------
elif st.session_state.current_page == "Show Model Selection":
    st.header("🤖 Model Selection")

    col1, col2 = st.columns(2)
    with col1:
        selected_model = st.selectbox("Select Regression Model", MODEL_NAMES)
    with col2:
        if selected_model == "Polynomial Regression":
            degree = st.slider("Select Polynomial Degree", 2, 5, 2)
        else:
            degree = 2

    test_size = st.slider(
        "Select Test Size for Model Evaluation",
        min_value=0.10,
        max_value=0.40,
        value=0.20,
        step=0.05
    )
    random_state = st.number_input(
        "Select Random State",
        min_value=0,
        max_value=1000,
        value=42,
        step=1,
        key="model_random_state"
    )

    if st.button("Train Selected Model", type="primary"):
        with st.spinner("Training model... Please wait."):
            X_train, X_test, y_train, y_test = split_data(
                test_size=test_size,
                random_state=int(random_state)
            )

            model = create_model(selected_model, degree)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            st.session_state.trained_model = model
            st.session_state.trained_model_name = selected_model
            st.session_state.model_metrics = {
                "MSE": mse,
                "RMSE": rmse,
                "MAE": mae,
                "R2 Score": r2
            }
            st.session_state.model_test_size = test_size

        st.success(f"{selected_model} trained successfully!")

    if "trained_model" in st.session_state:
        st.markdown("---")
        st.subheader(f"Model Performance: {st.session_state.trained_model_name}")

        metrics = st.session_state.model_metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MSE", f"{metrics['MSE']:,.2f}")
        col2.metric("RMSE", f"{metrics['RMSE']:,.2f}")
        col3.metric("MAE", f"{metrics['MAE']:,.2f}")
        col4.metric("R² Score", f"{metrics['R2 Score']:.4f}")

        model_bundle = {
            "model": st.session_state.trained_model,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "model_name": st.session_state.trained_model_name
        }

        model_bytes = pickle.dumps(model_bundle)
        st.download_button(
            label="⬇️ Download Trained Model",
            data=model_bytes,
            file_name="laptop_price_prediction_model.pkl",
            mime="application/octet-stream"
        )


# ---------------------------------------------------------
# 5. Prediction
# ---------------------------------------------------------
elif st.session_state.current_page == "Prediction":
    st.header("🔮 Laptop Price Prediction")

    if "trained_model" not in st.session_state:
        st.warning("Please go to Model Selection, train a model, and then return here.")
    else:
        model_bundle = {
            "model": st.session_state.trained_model,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "model_name": st.session_state.trained_model_name
        }

        model_bytes = pickle.dumps(model_bundle)
        st.download_button(
            label="⬇️ Download Trained Model",
            data=model_bytes,
            file_name="laptop_price_prediction_model.pkl",
            mime="application/octet-stream"
        )

        st.markdown("---")
        st.subheader("Enter Laptop Details")

        brand_options = sorted(df["Brand"].dropna().unique().tolist())

        col1, col2 = st.columns(2)

        with col1:
            brand = st.selectbox("Brand", brand_options)
            processor_speed = st.number_input(
                "Processor Speed",
                min_value=float(df["Processor_Speed"].min()),
                max_value=float(df["Processor_Speed"].max()),
                value=float(df["Processor_Speed"].median()),
                step=0.1
            )
            ram_size = st.selectbox(
                "RAM Size (GB)",
                sorted(df["RAM_Size"].unique().tolist())
            )

        with col2:
            storage_capacity = st.selectbox(
                "Storage Capacity (GB)",
                sorted(df["Storage_Capacity"].unique().tolist())
            )
            screen_size = st.number_input(
                "Screen Size (inches)",
                min_value=float(df["Screen_Size"].min()),
                max_value=float(df["Screen_Size"].max()),
                value=float(df["Screen_Size"].median()),
                step=0.1
            )
            weight = st.number_input(
                "Weight (kg)",
                min_value=float(df["Weight"].min()),
                max_value=float(df["Weight"].max()),
                value=float(df["Weight"].median()),
                step=0.1
            )

        if st.button("Predict Laptop Price", type="primary"):
            input_data = pd.DataFrame([{
                "Brand": brand,
                "Processor_Speed": processor_speed,
                "RAM_Size": ram_size,
                "Storage_Capacity": storage_capacity,
                "Screen_Size": screen_size,
                "Weight": weight
            }])

            predicted_price = st.session_state.trained_model.predict(input_data)[0]

            st.success("Prediction completed successfully!")
            st.metric(
                "Estimated Laptop Price",
                f"{predicted_price:,.2f}"
            )

            st.caption(
                f"Prediction generated using "
                f"{st.session_state.trained_model_name}."
            )
