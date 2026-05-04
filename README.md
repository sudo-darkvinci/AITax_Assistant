# AITax Assistant

**Explainable and Fair AI Nudges for Personalised Australian Tax Deductions**

An end-to-end Research Project Part B system that predicts high-risk tax deduction claims, delivers occupation-specific nudges, provides clear SHAP explanations, ensures fairness, and runs as a user-friendly web prototype.

### Features
- Hot-pluggable data loader (real government CSV/Excel files first, fallback to 50,000 synthetic records)
- Synthetic data calibrated to official ATO occupation averages
- XGBoost model with hyperparameter tuning (GridSearchCV)
- SHAP explainability (global summary + interactive waterfall plots)
- AIF360 fairness audits on gender and income brackets
- Occupation-specific behavioural nudges
- Fully functional Streamlit web prototype

### Research Project
-  Adelaide University
- Built on top of Part A scoping and literature review
- Directly addresses supervisor feedback: working prototype, user simulation, hyperparameter tuning, and hot-pluggable data

### How to Run

**Option 1: Run the Streamlit Prototype (Recommended)**
```bash
pip install -r requirements.txt
streamlit run app.py
