# 🔥 Fire Classification App: Machine Learning Deployment on Streamlit

This is a machine learning-based web application designed to classify fire incidents. The application uses historical MODIS satellite data and predicts the type of fire (such as forest fire, agricultural fire, etc.) based on user input parameters.

The primary goal of this project is to deploy a trained ML model through an interactive and easy-to-use Streamlit interface, making it accessible to anyone.

## 🚀 Live Demo

This application is deployed live on Streamlit Cloud. You can access it directly here:

👉 **[https://fire-classification.streamlit.app/]** 👈

## ✨ Features

* **Data Input:** Users can input various geographical and environmental parameters like **Temperature**, **Brightness**, and **Scan Angle**.
* **ML Prediction:** Uses a trained **scikit-learn** model (`best_fire_detection_model.pkl`) to predict the fire type in real-time.
* **Interactive Interface:** A simple and appealing interface for data input and result display using Streamlit.
* **Scalar Handling:** Input data is normalized using a **StandardScaler** (`scaler.pkl`) before being passed to the model, which is crucial for the model's accuracy.

## 📁 Project Structure

    . 
    ├── .streamlit/ 
    │   └── config.toml            # Streamlit configuration for Python version (3.12) 
    ├── app.py                    # Streamlit main application code 
    ├── requirements.txt          # All required Python libraries 
    ├── best_fire_detection_model.pkl  # Trained Machine Learning model (Managed by Git LFS) 
    ├── scaler.pkl                # MinMaxScaler object for data scaling (Managed by Git LFS) 
    ├── modis_2021_India.csv      # Historical data file 1 
    ├── modis_2022_India.csv      # Historical data file 2 
    └── modis_2023_India.csv      # Historical data file 3

## ⚙️ Local Setup

If you wish to run this app locally:

### 1. Install Dependencies

Ensure you have **Python 3.12** or a compatible version installed.

```bash
# Install all required libraries
pip install -r requirements.txt
```

### 2. Run the Streamlit App
```bash
streamlit run app.py
```

Your browser will automatically open and load the app at http://localhost:8501.

🔑 Key Technical Stack

Platform: Streamlit  
Programming Language: Python  
Data Science Libraries: scikit-learn, pandas, numpy  
Model Deployment: Streamlit Cloud  
Version Control: Git, Git LFS (for large model files)  

💡 Contributing

Feel free to submit a Pull Request if you would like to contribute improvements to this project.
