# No-Code ML Platform

A user-friendly, no-code machine learning platform built with Streamlit and AutoGluon. This application allows users to perform automated machine learning tasks on tabular, multimodal, and timeseries data without writing any code.

## Features

- **Tabular ML**: Automated model training and prediction for structured data
- **Multimodal ML**: Support for datasets with images and text
- **Timeseries Forecasting**: Time series analysis and prediction
- **Data Preprocessing**: Built-in tools for handling missing values, encoding, scaling, and feature engineering
- **Advanced Customization**: Fine-tune model parameters and training options
- **Model Management**: Save, load, and download trained models
- **Interactive UI**: Streamlit-based interface for easy data upload and result visualization

## Requirements

- Python 3.12 or higher
- UV package manager (for fast dependency management)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Jainam1673/No-Code-ML-Platform.git
   cd No-Code-ML-Platform
   ```

2. Install dependencies using UV:
   ```bash
   uv pip install -e .
   ```

   This will install all required packages including Streamlit, AutoGluon, and Pandas.

## Usage

Run the application with Streamlit:

```bash
streamlit run main.py
```

Open your browser and navigate to the provided URL (usually `http://localhost:8501`).

### How to Use

1. **Upload Data**: Use the file uploader to load your dataset (CSV, Excel, etc.)
2. **Select Task**: Choose between Tabular, Multimodal, or Timeseries tasks
3. **Preprocess Data**: Apply preprocessing steps as needed
4. **Train Model**: Configure training parameters and start automated model training
5. **Evaluate & Predict**: View model performance and make predictions on new data
6. **Download Results**: Save trained models and prediction results

## Project Structure

- `main.py`: Main application file containing all the Streamlit code
- `pyproject.toml`: Project configuration and dependencies
- `README.md`: This file

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.