# 🏥 Medical Insurance Cost Prediction & Dimensionality Reduction Analysis

An end-to-end data science and machine learning project performing Exploratory Data Analysis (EDA), missing value imputation, feature engineering, and dimensionality reduction techniques (**PCA** & **LDA**) on medical insurance policyholder data. The interactive analysis dashboard is deployed using **Streamlit**.

---

## 📌 Project Overview
Health insurance pricing relies heavily on accurately quantifying individual risk profiles to forecast expected medical expenditures. This project demonstrates:
* **Data Auditing & Cleaning**: Handling missing values using median/mode imputation, fixing typographical errors, and capping out-of-range domain outliers (e.g., `children > 5`).
* **Feature Engineering**: Creating clinical BMI risk categories (`underweight`, `healthy`, `overweight`, `obese`) and standardizing feature spaces.
* **Dimensionality Reduction**: Comparative analysis between unsupervised **Principal Component Analysis (PCA)** and supervised **Linear Discriminant Analysis (LDA)** to isolate high-risk policyholder clusters (e.g., smoking habits combined with high BMI).
* **Interactive Web App**: A user-friendly Streamlit dashboard to explore dataset distributions, PCA variance ratios, interactive 2D/3D scatter plots, and LDA linear decision boundaries.

---

## 🛠️ Project Structure

```text
├── app.py                  # Main Streamlit web application
├── Medical_Insurance.csv   # Dataset file
├── requirements.txt        # Python package dependencies
├── README.md               # Project documentation
└── .streamlit/
    └── config.toml         # Custom Streamlit styling / layout settings
```

---

## 💻 Local Setup & Installation

Follow these steps to run the project locally on your machine:

### 1. Prerequisites
Ensure you have **Python 3.8+** installed on your system. You can verify your installation by running:
```bash
python --version
```

### 2. Clone the Repository
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

### 3. Set Up a Virtual Environment (Recommended)
* **On macOS/Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
* **On Windows:**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

### 4. Install Dependencies
Install all required Python libraries using `pip`:
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Streamlit App Locally

Launch the Streamlit web application using the command below:
```bash
streamlit run app.py
```
Once executed, the application will automatically open in your default browser at `http://localhost:8501`.

---

## ☁️ How to Deploy on Streamlit Community Cloud

Deploying your Streamlit app directly from GitHub is free and takes less than 2 minutes:

1. **Push your code to GitHub**:
   Ensure `app.py`, `requirements.txt`, and your dataset (`Medical_Insurance.csv`) are committed and pushed to your GitHub repository.
2. **Sign in to Streamlit**:
   Go to [share.streamlit.io](https://share.streamlit.io/) and log in using your GitHub account.
3. **Deploy App**:
   * Click the **"New app"** button.
   * Select your GitHub repository, branch (`main` or `master`), and specify the main file path as `app.py`.
   * Click **"Deploy!"**. Your app will be live with a shareable public URL!

---

## 📊 Summary of Findings & Key Takeaways
1. **Smoker & BMI Interaction**: Smoking status combined with elevated BMI (>30) is the primary driver of extreme medical insurance claims.
2. **PCA vs. LDA**:
   * **PCA** (Unsupervised) captures global variance across continuous demographic variables but does not inherently group risk categories.
   * **LDA** (Supervised) maximizes class separability between high-cost and low-cost policyholders along the `smoker` boundary.

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
