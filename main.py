import streamlit as st
import pandas as pd
from autogluon.tabular import TabularPredictor
from autogluon.multimodal import MultiModalPredictor
from autogluon.timeseries import TimeSeriesPredictor
import os
import shutil
import tempfile
import numpy as np

# Page config
st.set_page_config(
    page_title="No Code ML Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'tabular_predictor' not in st.session_state:
    st.session_state.tabular_predictor = None
if 'multimodal_predictor' not in st.session_state:
    st.session_state.multimodal_predictor = None
if 'timeseries_predictor' not in st.session_state:
    st.session_state.timeseries_predictor = None
if 'tabular_df' not in st.session_state:
    st.session_state.tabular_df = None
if 'multimodal_df' not in st.session_state:
    st.session_state.multimodal_df = None
if 'timeseries_df' not in st.session_state:
    st.session_state.timeseries_df = None

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_data
def preprocess_data(df, fill_missing, encode_categorical="None", scale_numerical="None"):
    df = df.copy()
    if fill_missing == "Drop rows":
        df = df.dropna()
    elif fill_missing in ["Fill with mean", "Fill with median"]:
        for col in df.select_dtypes(include=[np.number]).columns:
            if fill_missing == "Fill with mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif fill_missing == "Fill with median":
                df[col].fillna(df[col].median(), inplace=True)
    elif fill_missing == "Fill with mode":
        for col in df.select_dtypes(include=['object']).columns:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else '', inplace=True)
    
    # Encode categorical
    if encode_categorical == "Label Encoding":
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = le.fit_transform(df[col])
    elif encode_categorical == "One-Hot Encoding":
        df = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns)
    
    # Scale numerical
    if scale_numerical == "Standard Scaler":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = scaler.fit_transform(df[num_cols])
    elif scale_numerical == "Min-Max Scaler":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = scaler.fit_transform(df[num_cols])
    
    return df

st.set_page_config(page_title="No Code ML Platform", page_icon="🤖", layout="wide")

st.title("🤖 No Code ML Platform with AutoGluon")

st.markdown("""
Welcome to the ultimate **no-code machine learning platform** powered by **AutoGluon**!
Choose your task type and let AutoGluon handle the rest with **lightning-fast** automation.
""")

# Sidebar for global settings
st.sidebar.header("⚙️ Global Settings")
time_limit = st.sidebar.slider("⏱️ Training Time Limit (seconds)", min_value=60, max_value=3600, value=300, step=60)
presets = st.sidebar.selectbox("🎯 Presets", ["best_quality", "high_quality", "good_quality", "medium_quality", "auto"])

# Main tabs
tab1, tab2, tab3 = st.tabs(["📊 Tabular Data", "🖼️ Multimodal", "📈 Time Series"])

# Tabular Tab
with tab1:
    st.header("Tabular Data Tasks")
    st.write("Upload your CSV dataset for classification or regression tasks.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="tabular")

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.session_state.tabular_df = df
        st.write("Data Preview:")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.dataframe(df.head(), use_container_width=True)
        with col2:
            st.write("**Data Info:**")
            st.write(f"Rows: {len(df)}")
            st.write(f"Columns: {len(df.columns)}")
            st.write(f"Missing values: {df.isnull().sum().sum()}")

        # Data Preprocessing
        with st.expander("Data Preprocessing"):
            fill_missing = st.selectbox("Handle Missing Values", ["None", "Drop rows", "Fill with mean", "Fill with median", "Fill with mode"], key="tabular_preprocess")
            encode_categorical = st.selectbox("Encode Categorical Variables", ["None", "Label Encoding", "One-Hot Encoding"], key="tabular_encode")
            scale_numerical = st.selectbox("Scale Numerical Variables", ["None", "Standard Scaler", "Min-Max Scaler"], key="tabular_scale")
            if fill_missing != "None" or encode_categorical != "None" or scale_numerical != "None":
                df = preprocess_data(df, fill_missing, encode_categorical, scale_numerical)
                st.session_state.tabular_df = df
                st.write("Data after preprocessing:")
                st.dataframe(df.head(), use_container_width=True)

        columns = df.columns.tolist()
        target_column = st.selectbox("Select the target column", columns, key="tabular_target")

        problem_type_options = ["auto", "binary", "multiclass", "regression"]
        problem_type = st.selectbox("Select the problem type", problem_type_options, key="tabular_problem")

        # Advanced options
        with st.expander("Advanced Options"):
            eval_metric = st.selectbox("Evaluation Metric", ["auto", "accuracy", "f1", "precision", "recall", "roc_auc", "log_loss", "mae", "mse", "r2"], key="tabular_metric")
            hyperparameter_tune = st.checkbox("Enable Hyperparameter Tuning", key="tabular_tune")
            excluded_model_types = st.multiselect("Exclude Model Types", ["KNN", "GBM", "CAT", "XGB", "RF", "XT", "LR", "NN_TORCH", "FASTAI"], key="tabular_exclude")
            custom_hyperparameters = st.text_area("Custom Hyperparameters (JSON format)", placeholder='{"GBM": {"num_boost_round": 100}}', key="tabular_hyper")

        if st.button("🚀 Train Model", key="tabular_train"):
            df = st.session_state.tabular_df or df
            with st.spinner("Training model... This may take a while."):
                # Parse custom hyperparameters
                hyperparams = None
                if custom_hyperparameters:
                    try:
                        import json
                        hyperparams = json.loads(custom_hyperparameters)
                    except:
                        st.error("Invalid JSON for custom hyperparameters. Using defaults.")

                predictor = TabularPredictor(
                    label=target_column,
                    problem_type=problem_type if problem_type != "auto" else None,
                    eval_metric=eval_metric if eval_metric != "auto" else None,
                    path=tempfile.mkdtemp()
                )
                fit_kwargs = {
                    'time_limit': time_limit,
                    'presets': presets,
                }
                if hyperparameter_tune:
                    fit_kwargs['hyperparameter_tune_kwargs'] = 'auto'
                if excluded_model_types:
                    fit_kwargs['excluded_model_types'] = excluded_model_types
                if hyperparams:
                    fit_kwargs['hyperparameters'] = hyperparams

                predictor.fit(df, **fit_kwargs)
                st.session_state.tabular_predictor = predictor

            st.success("✅ Model trained successfully!")

            # Display results in columns
            col1, col2 = st.columns(2)
            with col1:
                st.write("**🏆 Model Leaderboard:**")
                leaderboard = predictor.leaderboard()
                st.dataframe(leaderboard, use_container_width=True)

            with col2:
                try:
                    importance = predictor.feature_importance(df)
                    st.write("**📊 Feature Importance:**")
                    st.dataframe(importance, use_container_width=True)
                except:
                    st.write("Feature importance not available for this model.")

            # Sample predictions
            st.write("**🔮 Sample Predictions on training data:**")
            predictions = predictor.predict(df.head(10))
            st.dataframe(predictions, use_container_width=True)

            # Save and download
            model_path = "trained_tabular_model"
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
            predictor.save(model_path)
            shutil.make_archive(model_path, 'zip', model_path)
            with open(f"{model_path}.zip", "rb") as f:
                st.download_button("📥 Download Model", f, file_name="trained_tabular_model.zip", key="tabular_download", use_container_width=True)

            # Batch predictions
            st.write("### 🔮 Batch Predictions")
            test_file = st.file_uploader("Upload test data CSV for predictions", type="csv", key="tabular_test")
            if test_file is not None:
                test_df = load_data(test_file)
                st.write("Test Data Preview:")
                st.dataframe(test_df.head(), use_container_width=True)
                if st.button("🔍 Make Predictions", key="tabular_predict"):
                    predictions = predictor.predict(test_df)
                    st.write("Predictions:")
                    st.dataframe(predictions, use_container_width=True)
                    # Download predictions
                    csv = predictions.to_csv(index=False)
                    st.download_button("📥 Download Predictions", csv, file_name="predictions.csv", mime="text/csv", key="tabular_pred_download", use_container_width=True)

    else:
        st.write("Please upload a CSV file to get started.")

# Multimodal Tab
with tab2:
    st.header("Multimodal Tasks")
    st.write("Handle tabular data with text and images using multimodal learning.")

    uploaded_csv = st.file_uploader("Choose a CSV file", type="csv", key="multimodal_csv")
    uploaded_images = st.file_uploader("Choose a ZIP file with images", type="zip", key="multimodal_images")

    if uploaded_csv is not None:
        df = load_data(uploaded_csv)
        st.write("Data Preview:")
        st.dataframe(df.head())

        columns = df.columns.tolist()
        target_column = st.selectbox("Select the target column", columns, key="multimodal_target")

        text_columns = st.multiselect("Select text columns", columns, key="multimodal_text")
        image_column = st.selectbox("Select image column (paths)", ["None"] + columns, key="multimodal_image")

        problem_type_options = ["auto", "binary", "multiclass", "regression"]
        problem_type = st.selectbox("Select the problem type", problem_type_options, key="multimodal_problem")

        eval_metric_options = ["auto", "accuracy", "f1", "precision", "recall", "roc_auc", "log_loss", "mae", "mse", "r2"]
        eval_metric = st.selectbox("Evaluation Metric", eval_metric_options, key="multimodal_metric")

        if st.button("Train Model", key="multimodal_train"):
            with st.spinner("Training multimodal model... This may take a while."):
                # Extract images if uploaded
                image_path = None
                if uploaded_images is not None and image_column != "None":
                    image_path = tempfile.mkdtemp()
                    zip_path = tempfile.mktemp(suffix=".zip")
                    with open(zip_path, "wb") as f:
                        f.write(uploaded_images.getvalue())
                    shutil.unpack_archive(zip_path, image_path)
                    # Assume images are in subfolder or adjust paths
                    df[image_column] = df[image_column].apply(lambda x: os.path.join(image_path, x) if pd.notna(x) else x)

                predictor = MultiModalPredictor(
                    label=target_column,
                    problem_type=problem_type if problem_type != "auto" else None,
                    eval_metric=eval_metric if eval_metric != "auto" else None,
                    path=tempfile.mkdtemp()
                )

                predictor.fit(
                    df,
                    time_limit=time_limit,
                    presets=presets,
                    column_types={"text": text_columns, "image": [image_column] if image_column != "None" else []}
                )

            st.success("Multimodal model trained successfully!")

            # Display leaderboard
            st.write("Model Leaderboard:")
            leaderboard = predictor.leaderboard()
            st.dataframe(leaderboard)

            # Save model
            model_path = "trained_multimodal_model"
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
            predictor.save(model_path)
            st.write(f"Model saved to {model_path}")

            # Download model
            shutil.make_archive(model_path, 'zip', model_path)
            with open(f"{model_path}.zip", "rb") as f:
                st.download_button("Download Model", f, file_name="trained_multimodal_model.zip", key="multimodal_download")

            # Batch predictions
            st.write("### Batch Predictions")
            test_file = st.file_uploader("Upload test data CSV for predictions", type="csv", key="multimodal_test")
            if test_file is not None:
                test_df = pd.read_csv(test_file)
                st.write("Test Data Preview:")
                st.dataframe(test_df.head())
                if st.button("Make Predictions", key="multimodal_predict"):
                    predictions = predictor.predict(test_df)
                    st.write("Predictions:")
                    st.dataframe(predictions)
                    # Download predictions
                    csv = predictions.to_csv(index=False)
                    st.download_button("Download Predictions", csv, file_name="predictions.csv", mime="text/csv", key="multimodal_pred_download")

    else:
        st.write("Please upload a CSV file to get started.")

# Time Series Tab
with tab3:
    st.header("Time Series Forecasting")
    st.write("Upload your time series data for forecasting tasks.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="timeseries")

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df.head())

        columns = df.columns.tolist()
        timestamp_column = st.selectbox("Select the timestamp column", columns, key="timeseries_timestamp")
        target_column = st.selectbox("Select the target column", columns, key="timeseries_target")
        item_id_column = st.selectbox("Select the item ID column (optional)", ["None"] + columns, key="timeseries_item")

        prediction_length = st.slider("Prediction Length", min_value=1, max_value=100, value=10, key="timeseries_length")

        eval_metric_options = ["auto", "MAE", "MSE", "RMSE", "MAPE", "SMAPE", "MASE"]
        eval_metric = st.selectbox("Evaluation Metric", eval_metric_options, key="timeseries_metric")

        if st.button("Train and Forecast", key="timeseries_train"):
            with st.spinner("Training time series model... This may take a while."):
                # Prepare data
                df[timestamp_column] = pd.to_datetime(df[timestamp_column])
                df = df.sort_values([item_id_column if item_id_column != "None" else timestamp_column, timestamp_column])

                from autogluon.timeseries import TimeSeriesDataFrame
                if item_id_column != "None":
                    train_data = TimeSeriesDataFrame.from_data_frame(
                        df,
                        id_column=item_id_column,
                        timestamp_column=timestamp_column
                    )
                else:
                    # Single series, add dummy id
                    df = df.copy()
                    df['item_id'] = 'series_1'
                    train_data = TimeSeriesDataFrame.from_data_frame(
                        df,
                        id_column='item_id',
                        timestamp_column=timestamp_column
                    )

                predictor = TimeSeriesPredictor(
                    prediction_length=prediction_length,
                    target=target_column,
                    eval_metric=eval_metric if eval_metric != "auto" else None,
                    path=tempfile.mkdtemp()
                )

                predictor.fit(
                    train_data,
                    time_limit=time_limit,
                    presets=presets
                )

            st.success("Time series model trained successfully!")

            # Display leaderboard
            st.write("Model Leaderboard:")
            leaderboard = predictor.leaderboard()
            st.dataframe(leaderboard)

            # Make predictions
            st.write(f"Forecasting next {prediction_length} periods:")
            predictions = predictor.predict(df)
            st.dataframe(predictions.head())

            # Save model
            model_path = "trained_timeseries_model"
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
            predictor.save(model_path)
            st.write(f"Model saved to {model_path}")

            # Download model
            shutil.make_archive(model_path, 'zip', model_path)
            with open(f"{model_path}.zip", "rb") as f:
                st.download_button("Download Model", f, file_name="trained_timeseries_model.zip", key="timeseries_download")

    else:
        st.write("Please upload a CSV file to get started.")
