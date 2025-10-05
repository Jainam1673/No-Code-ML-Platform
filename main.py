"""
No-Code ML Platform with AutoGluon

A comprehensive machine learning platform that leverages AutoGluon for
automated model training and prediction across tabular, multimodal, and
time series data.
"""

import json
import os
import shutil
import tempfile
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st
from autogluon.multimodal import MultiModalPredictor
from autogluon.tabular import TabularPredictor
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


# Constants
DEFAULT_TIME_LIMIT = 300
DEFAULT_PRESETS = "medium"
MAX_TIME_LIMIT = 3600
MIN_TIME_LIMIT = 60
TIME_STEP = 60
PAGE_TITLE = "No Code ML Platform"
PAGE_ICON = "🤖"
LAYOUT = "wide"
SIDEBAR_STATE = "expanded"

# Session state keys
TABULAR_PREDICTOR_KEY = "tabular_predictor"
MULTIMODAL_PREDICTOR_KEY = "multimodal_predictor"
TIMESERIES_PREDICTOR_KEY = "timeseries_predictor"
TABULAR_DF_KEY = "tabular_df"
MULTIMODAL_DF_KEY = "multimodal_df"
TIMESERIES_DF_KEY = "timeseries_df"


class DataHandler:
    """Handles data loading and basic operations."""

    @staticmethod
    @st.cache_data
    def load_data(uploaded_file) -> pd.DataFrame:
        """Load CSV data from uploaded file."""
        return pd.read_csv(uploaded_file)

    @staticmethod
    def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic information about the dataset."""
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "missing_values": df.isnull().sum().sum(),
            "dtypes": df.dtypes.to_dict()
        }


class Preprocessor:
    """Handles data preprocessing operations."""

    def __init__(self):
        self.scalers = {}

    @st.cache_data
    def preprocess(
        self,
        df: pd.DataFrame,
        fill_missing: str = "None",
        encode_categorical: str = "None",
        scale_numerical: str = "None"
    ) -> pd.DataFrame:
        """Apply preprocessing transformations to the dataframe."""
        df = df.copy()

        # Handle missing values
        df = self._handle_missing_values(df, fill_missing)

        # Encode categorical variables
        df = self._encode_categorical(df, encode_categorical)

        # Scale numerical variables
        df = self._scale_numerical(df, scale_numerical)

        return df

    def _handle_missing_values(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """Handle missing values in the dataframe."""
        if method == "Drop rows":
            return df.dropna()
        elif method in ["Fill with mean", "Fill with median"]:
            for col in df.select_dtypes(include=[np.number]).columns:
                if method == "Fill with mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif method == "Fill with median":
                    df[col] = df[col].fillna(df[col].median())
        elif method == "Fill with mode":
            for col in df.select_dtypes(include=['object']).columns:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])
        return df

    def _encode_categorical(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """Encode categorical variables."""
        if method == "Label Encoding":
            le = LabelEncoder()
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = le.fit_transform(df[col].astype(str))
        elif method == "One-Hot Encoding":
            df = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns)
        return df

    def _scale_numerical(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """Scale numerical variables."""
        if method == "Standard Scaler":
            scaler = StandardScaler()
            num_cols = df.select_dtypes(include=[np.number]).columns
            df[num_cols] = scaler.fit_transform(df[num_cols])
        elif method == "Min-Max Scaler":
            scaler = MinMaxScaler()
            num_cols = df.select_dtypes(include=[np.number]).columns
            df[num_cols] = scaler.fit_transform(df[num_cols])
        return df


class BasePredictorHandler:
    """Base class for ML predictor handlers."""

    def __init__(self, predictor_key: str, df_key: str):
        self.predictor_key = predictor_key
        self.df_key = df_key
        self.preprocessor = Preprocessor()

    def get_predictor(self):
        """Get the predictor from session state."""
        return st.session_state.get(self.predictor_key)

    def set_predictor(self, predictor):
        """Set the predictor in session state."""
        st.session_state[self.predictor_key] = predictor

    def get_dataframe(self):
        """Get the dataframe from session state."""
        return st.session_state.get(self.df_key)

    def set_dataframe(self, df: pd.DataFrame):
        """Set the dataframe in session state."""
        st.session_state[self.df_key] = df

    def save_model(self, predictor, model_name: str):
        """Save and create downloadable model archive."""
        model_path = f"trained_{model_name}_model"
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        predictor.save(model_path)
        shutil.make_archive(model_path, 'zip', model_path)
        return model_path

    def download_button(self, label: str, data, filename: str, key: str):
        """Create a download button."""
        st.download_button(label, data, filename=filename, key=key)


class TabularHandler(BasePredictorHandler):
    """Handler for tabular data tasks."""

    def __init__(self):
        super().__init__(TABULAR_PREDICTOR_KEY, TABULAR_DF_KEY)

    def render_ui(self):
        """Render the tabular data UI."""
        st.header("Tabular Data Tasks")
        st.write("Upload your CSV dataset for classification or regression tasks.")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="tabular")

        if uploaded_file is not None:
            df = DataHandler.load_data(uploaded_file)
            self.set_dataframe(df)

            # Data preview and info
            self._render_data_preview(df)

            # Preprocessing
            df = self._render_preprocessing(df)
            self.set_dataframe(df)

            # Model configuration
            config = self._get_model_config(df.columns.tolist())

            if st.button("🚀 Train Model", key="tabular_train"):
                self._train_model(df, config)

            # Display results if model exists
            predictor = self.get_predictor()
            if predictor:
                self._display_results(predictor, df)

    def _render_data_preview(self, df: pd.DataFrame):
        """Render data preview and information."""
        st.write("Data Preview:")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.dataframe(df.head(), use_container_width=True)
        with col2:
            info = DataHandler.get_data_info(df)
            st.write("**Data Info:**")
            st.write(f"Rows: {info['rows']}")
            st.write(f"Columns: {info['columns']}")
            st.write(f"Missing values: {info['missing_values']}")

    def _render_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Render preprocessing options and apply them."""
        with st.expander("Data Preprocessing"):
            fill_missing = st.selectbox(
                "Handle Missing Values",
                ["None", "Drop rows", "Fill with mean", "Fill with median", "Fill with mode"],
                key="tabular_preprocess"
            )
            encode_categorical = st.selectbox(
                "Encode Categorical Variables",
                ["None", "Label Encoding", "One-Hot Encoding"],
                key="tabular_encode"
            )
            scale_numerical = st.selectbox(
                "Scale Numerical Variables",
                ["None", "Standard Scaler", "Min-Max Scaler"],
                key="tabular_scale"
            )

            if any([fill_missing != "None", encode_categorical != "None", scale_numerical != "None"]):
                df = self.preprocessor.preprocess(df, fill_missing, encode_categorical, scale_numerical)
                st.write("Data after preprocessing:")
                st.dataframe(df.head(), use_container_width=True)

        return df

    def _get_model_config(self, columns: List[str]) -> Dict[str, Any]:
        """Get model configuration from UI."""
        target_column = st.selectbox("Select the target column", columns, key="tabular_target")

        problem_type = st.selectbox(
            "Select the problem type",
            ["auto", "binary", "multiclass", "regression"],
            key="tabular_problem"
        )

        # Advanced options
        with st.expander("Advanced Options"):
            eval_metric = st.selectbox(
                "Evaluation Metric",
                ["auto", "accuracy", "f1", "precision", "recall", "roc_auc", "log_loss", "mae", "mse", "r2"],
                key="tabular_metric"
            )
            hyperparameter_tune = st.checkbox("Enable Hyperparameter Tuning", key="tabular_tune")
            excluded_model_types = st.multiselect(
                "Exclude Model Types",
                ["KNN", "GBM", "CAT", "XGB", "RF", "XT", "LR", "NN_TORCH", "FASTAI"],
                key="tabular_exclude"
            )
            custom_hyperparameters = st.text_area(
                "Custom Hyperparameters (JSON format)",
                placeholder='{"GBM": {"num_boost_round": 100}}',
                key="tabular_hyper"
            )

        return {
            "target_column": target_column,
            "problem_type": problem_type,
            "eval_metric": eval_metric,
            "hyperparameter_tune": hyperparameter_tune,
            "excluded_model_types": excluded_model_types,
            "custom_hyperparameters": custom_hyperparameters
        }

    def _train_model(self, df: pd.DataFrame, config: Dict[str, Any]):
        """Train the tabular model."""
        with st.spinner("Training model... This may take a while."):
            # Parse custom hyperparameters
            hyperparams = None
            if config["custom_hyperparameters"]:
                try:
                    hyperparams = json.loads(config["custom_hyperparameters"])
                except json.JSONDecodeError:
                    st.error("Invalid JSON for custom hyperparameters. Using defaults.")

            predictor = TabularPredictor(
                label=config["target_column"],
                problem_type=config["problem_type"] if config["problem_type"] != "auto" else None,
                eval_metric=config["eval_metric"] if config["eval_metric"] != "auto" else None,
                path=tempfile.mkdtemp()
            )

            fit_kwargs = {
                'time_limit': st.session_state.get('time_limit', DEFAULT_TIME_LIMIT),
                'presets': st.session_state.get('presets', DEFAULT_PRESETS),
            }

            if config["hyperparameter_tune"]:
                fit_kwargs['hyperparameter_tune_kwargs'] = 'auto'
            if config["excluded_model_types"]:
                fit_kwargs['excluded_model_types'] = config["excluded_model_types"]
            if hyperparams:
                fit_kwargs['hyperparameters'] = hyperparams

            predictor.fit(df, **fit_kwargs)
            self.set_predictor(predictor)

        st.success("✅ Model trained successfully!")

    def _display_results(self, predictor, df: pd.DataFrame):
        """Display training results and provide prediction interface."""
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
            except Exception:
                st.write("Feature importance not available for this model.")

        # Sample predictions
        st.write("**🔮 Sample Predictions on training data:**")
        predictions = predictor.predict(df.head(10))
        st.dataframe(predictions, use_container_width=True)

        # Save and download
        model_path = self.save_model(predictor, "tabular")
        with open(f"{model_path}.zip", "rb") as f:
            self.download_button(
                "📥 Download Model",
                f,
                "trained_tabular_model.zip",
                "tabular_download"
            )

        # Batch predictions
        self._render_batch_predictions()

    def _render_batch_predictions(self):
        """Render batch prediction interface."""
        st.write("### 🔮 Batch Predictions")
        test_file = st.file_uploader("Upload test data CSV for predictions", type="csv", key="tabular_test")
        if test_file is not None:
            test_df = DataHandler.load_data(test_file)
            st.write("Test Data Preview:")
            st.dataframe(test_df.head(), use_container_width=True)
            if st.button("🔍 Make Predictions", key="tabular_predict"):
                predictor = self.get_predictor()
                predictions = predictor.predict(test_df)
                st.write("Predictions:")
                st.dataframe(predictions, use_container_width=True)
                # Download predictions
                csv = predictions.to_csv(index=False)
                self.download_button(
                    "📥 Download Predictions",
                    csv,
                    "predictions.csv",
                    "tabular_pred_download"
                )


class MultimodalHandler(BasePredictorHandler):
    """Handler for multimodal data tasks."""

    def __init__(self):
        super().__init__(MULTIMODAL_PREDICTOR_KEY, MULTIMODAL_DF_KEY)

    def render_ui(self):
        """Render the multimodal data UI."""
        st.header("Multimodal Tasks")
        st.write("Handle tabular data with text and images using multimodal learning.")

        uploaded_csv = st.file_uploader("Choose a CSV file", type="csv", key="multimodal_csv")
        uploaded_images = st.file_uploader("Choose a ZIP file with images", type="zip", key="multimodal_images")

        if uploaded_csv is not None:
            df = DataHandler.load_data(uploaded_csv)
            st.write("Data Preview:")
            st.dataframe(df.head())

            columns = df.columns.tolist()
            target_column = st.selectbox("Select the target column", columns, key="multimodal_target")

            text_columns = st.multiselect("Select text columns", columns, key="multimodal_text")
            image_column = st.selectbox("Select image column (paths)", ["None"] + columns, key="multimodal_image")

            problem_type = st.selectbox(
                "Select the problem type",
                ["auto", "binary", "multiclass", "regression"],
                key="multimodal_problem"
            )

            eval_metric = st.selectbox(
                "Evaluation Metric",
                ["auto", "accuracy", "f1", "precision", "recall", "roc_auc", "log_loss", "mae", "mse", "r2"],
                key="multimodal_metric"
            )

            # Advanced options
            with st.expander("Advanced Options"):
                time_limit = st.slider("Time Limit (seconds)", min_value=10, max_value=3600, value=600, key="multimodal_time")
                presets = st.selectbox("Presets", ["medium", "high", "best"], key="multimodal_presets")

            if st.button("Train Model", key="multimodal_train"):
                self._train_model(df, target_column, problem_type, eval_metric, text_columns, image_column, uploaded_images, time_limit, presets)

            # Display results if model exists
            predictor = self.get_predictor()
            if predictor:
                self._display_results(predictor)

    def _train_model(self, df, target_column, problem_type, eval_metric, text_columns, image_column, uploaded_images, time_limit, presets):
        """Train the multimodal model."""
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
                df = df.copy()
                df[image_column] = df[image_column].apply(
                    lambda x: os.path.join(image_path, x) if pd.notna(x) else x
                )

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

            self.set_predictor(predictor)

        st.success("Multimodal model trained successfully!")

    def _display_results(self, predictor):
        """Display training results."""
        st.write("Model Leaderboard:")
        leaderboard = predictor.leaderboard()
        st.dataframe(leaderboard)

        # Save model
        model_path = self.save_model(predictor, "multimodal")
        st.write(f"Model saved to {model_path}")

        # Download model
        with open(f"{model_path}.zip", "rb") as f:
            self.download_button(
                "Download Model",
                f,
                "trained_multimodal_model.zip",
                "multimodal_download"
            )

        # Batch predictions
        self._render_batch_predictions()

    def _render_batch_predictions(self):
        """Render batch prediction interface."""
        st.write("### Batch Predictions")
        test_file = st.file_uploader("Upload test data CSV for predictions", type="csv", key="multimodal_test")
        if test_file is not None:
            test_df = pd.read_csv(test_file)
            st.write("Test Data Preview:")
            st.dataframe(test_df.head())
            if st.button("Make Predictions", key="multimodal_predict"):
                predictor = self.get_predictor()
                predictions = predictor.predict(test_df)
                st.write("Predictions:")
                st.dataframe(predictions)
                # Download predictions
                csv = predictions.to_csv(index=False)
                self.download_button(
                    "Download Predictions",
                    csv,
                    "predictions.csv",
                    "multimodal_pred_download"
                )


class TimeSeriesHandler(BasePredictorHandler):
    """Handler for time series forecasting tasks."""

    def __init__(self):
        super().__init__(TIMESERIES_PREDICTOR_KEY, TIMESERIES_DF_KEY)

    def render_ui(self):
        """Render the time series UI."""
        st.header("Time Series Forecasting")
        st.write("Upload your time series data for forecasting tasks.")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="timeseries")

        if uploaded_file is not None:
            df = DataHandler.load_data(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(df.head())

            columns = df.columns.tolist()
            timestamp_column = st.selectbox("Select the timestamp column", columns, key="timeseries_timestamp")
            target_column = st.selectbox("Select the target column", columns, key="timeseries_target")
            item_id_column = st.selectbox("Select the item ID column (optional)", ["None"] + columns, key="timeseries_item")

            prediction_length = st.slider("Prediction Length", min_value=1, max_value=100, value=10, key="timeseries_length")

            eval_metric = st.selectbox(
                "Evaluation Metric",
                ["auto", "MAE", "MSE", "RMSE", "MAPE", "SMAPE", "MASE"],
                key="timeseries_metric"
            )

            # Advanced options
            with st.expander("Advanced Options"):
                time_limit = st.slider("Time Limit (seconds)", min_value=10, max_value=3600, value=600, key="timeseries_time")
                presets = st.selectbox("Presets", ["fast_training", "medium_quality", "high_quality", "best_quality"], key="timeseries_presets")
                use_chronos = st.checkbox("Use Chronos (Zero-shot)", key="timeseries_chronos")
                prediction_intervals = st.checkbox("Generate Prediction Intervals", key="timeseries_intervals")

            if st.button("Train and Forecast", key="timeseries_train"):
                self._train_model(df, timestamp_column, target_column, item_id_column, prediction_length, eval_metric, time_limit, presets, use_chronos, prediction_intervals)

            # Display results if model exists
            predictor = self.get_predictor()
            if predictor:
                self._display_results(predictor, df, prediction_length, prediction_intervals)

    def _train_model(self, df, timestamp_column, target_column, item_id_column, prediction_length, eval_metric, time_limit, presets, use_chronos, prediction_intervals):
        """Train the time series model."""
        with st.spinner("Training time series model... This may take a while."):
            # Prepare data
            df = df.copy()
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])
            df = df.sort_values([item_id_column if item_id_column != "None" else timestamp_column, timestamp_column])

            if item_id_column != "None":
                train_data = TimeSeriesDataFrame.from_data_frame(
                    df,
                    id_column=item_id_column,
                    timestamp_column=timestamp_column
                )
            else:
                # Single series, add dummy id
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

            fit_kwargs = {
                'time_limit': time_limit,
                'presets': presets
            }
            if use_chronos:
                fit_kwargs['hyperparameters'] = {'Chronos': {'model_path': 'amazon/chronos-t5-small'}}

            predictor.fit(train_data, **fit_kwargs)
            self.set_predictor(predictor)

        st.success("Time series model trained successfully!")

    def _display_results(self, predictor, df, prediction_length, prediction_intervals):
        """Display training results and forecasts."""
        st.write("Model Leaderboard:")
        leaderboard = predictor.leaderboard()
        st.dataframe(leaderboard)

        # Make predictions
        st.write(f"Forecasting next {prediction_length} periods:")
        predict_kwargs = {}
        if prediction_intervals:
            predict_kwargs['quantile_levels'] = [0.1, 0.5, 0.9]
        predictions = predictor.predict(df, **predict_kwargs)
        st.dataframe(predictions.head())

        # Save model
        model_path = self.save_model(predictor, "timeseries")
        st.write(f"Model saved to {model_path}")

        # Download model
        with open(f"{model_path}.zip", "rb") as f:
            self.download_button(
                "Download Model",
                f,
                "trained_timeseries_model.zip",
                "timeseries_download"
            )


class MLPlatform:
    """Main application class for the No-Code ML Platform."""

    def __init__(self):
        self.tabular_handler = TabularHandler()
        self.multimodal_handler = MultimodalHandler()
        self.timeseries_handler = TimeSeriesHandler()
        self.setup_page()
        self.initialize_session_state()

    def setup_page(self):
        """Configure the Streamlit page."""
        st.set_page_config(
            page_title=PAGE_TITLE,
            page_icon=PAGE_ICON,
            layout=LAYOUT,
            initial_sidebar_state=SIDEBAR_STATE
        )

    def initialize_session_state(self):
        """Initialize session state variables."""
        if TABULAR_PREDICTOR_KEY not in st.session_state:
            st.session_state[TABULAR_PREDICTOR_KEY] = None
        if MULTIMODAL_PREDICTOR_KEY not in st.session_state:
            st.session_state[MULTIMODAL_PREDICTOR_KEY] = None
        if TIMESERIES_PREDICTOR_KEY not in st.session_state:
            st.session_state[TIMESERIES_PREDICTOR_KEY] = None
        if TABULAR_DF_KEY not in st.session_state:
            st.session_state[TABULAR_DF_KEY] = None
        if MULTIMODAL_DF_KEY not in st.session_state:
            st.session_state[MULTIMODAL_DF_KEY] = None
        if TIMESERIES_DF_KEY not in st.session_state:
            st.session_state[TIMESERIES_DF_KEY] = None

    def render_sidebar(self):
        """Render the sidebar with global settings."""
        st.sidebar.header("⚙️ Global Settings")
        st.session_state['time_limit'] = st.sidebar.slider(
            "⏱️ Training Time Limit (seconds)",
            min_value=MIN_TIME_LIMIT,
            max_value=MAX_TIME_LIMIT,
            value=DEFAULT_TIME_LIMIT,
            step=TIME_STEP
        )
        st.session_state['presets'] = st.sidebar.selectbox(
            "🎯 Presets",
            ["extreme", "best", "high", "good", "medium", "auto"]
        )

    def render_main_content(self):
        """Render the main content area."""
        st.title("🤖 No Code ML Platform with AutoGluon")
        st.markdown("""
        Welcome to the ultimate **no-code machine learning platform** powered by **AutoGluon**!
        Choose your task type and let AutoGluon handle the rest with **lightning-fast** automation.
        """)

    def render_tabs(self):
        """Render the main tabs."""
        tab1, tab2, tab3 = st.tabs(["📊 Tabular Data", "🖼️ Multimodal", "📈 Time Series"])

        with tab1:
            self.tabular_handler.render_ui()

        with tab2:
            self.multimodal_handler.render_ui()

        with tab3:
            self.timeseries_handler.render_ui()

    def run(self):
        """Run the application."""
        self.render_sidebar()
        self.render_main_content()
        self.render_tabs()


def main():
    """Main entry point."""
    app = MLPlatform()
    app.run()


if __name__ == "__main__":
    main()
