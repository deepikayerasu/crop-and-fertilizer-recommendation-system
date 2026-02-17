Smart Agriculture Recommendation System

An end-to-end Machine Learning web application that recommends:
-Optimal Crop based on soil nutrients and environmental conditions
-Suitable Fertilizer based on soil type and crop selection

Built using multiple classification models with cross-validation and automatic best model selection.

Features
- Model comparison (Logistic Regression, KNN, Random Forest)
- Automatic best model selection
- Feature importance visualization
- Interactive Streamlit web interface



Tech Stack
- Python
- Pandas & NumPy
- Scikit-learn
- Matplotlib
- Streamlit


Run Locally

```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
