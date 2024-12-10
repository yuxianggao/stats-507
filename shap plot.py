### drawing shap plot



import shap

# Create SHAP explainer for Logistic Regression
feature_names = X_train.columns.tolist()

explainer = shap.LinearExplainer(log_model, X_train)

# # Compute SHAP values for the test set

# # Summary plot for feature importance

# Sample 1000 rows from the test set
X_test_sample = shap.sample(X_test, 1000)

# Compute SHAP values on the smaller dataset
shap_values_sample = explainer(X_test_sample)

# Summary plot
shap.summary_plot(shap_values_sample.values, X_test_sample, feature_names=feature_names, max_display=20)