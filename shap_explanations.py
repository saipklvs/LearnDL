# Attempting again using KernelExplainer with strict CPU mode to avoid GPU dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt
import os 

# Load the Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=42, n_estimators=100)
rf_clf.fit(X_train, y_train)

# Test the classifier
y_pred = rf_clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

## save the predictions in a temp predictions file
# Create the temp directory if it doesn't exist
storage_path_predictions = "temp/saved_predictions/"
os.makedirs(storage_path_predictions, exist_ok=True)

# Save the predictions to a CSV file
X_test = X_test.reset_index(drop=True)
combined_df = pd.concat([X_test, pd.Series(y_test, name="True Class"), pd.Series(y_pred, name="Predicted Class")], axis=1)
combined_df.to_csv(os.path.join(storage_path_predictions, "predictions.csv"), index=False)


# Use KernelExplainer for SHAP values computation
# KernelExplainer is purely CPU-based and does not require GPU dependencies
explainer = shap.KernelExplainer(rf_clf.predict_proba, X_train)  # Small sample for baseline
shap_values = explainer.shap_values(X_test)

print("SHAP Values Shape:")
print([shap_values[i].shape for i in range(len(shap_values))])

# # Create the temp directory if it doesn't exist
storage_path_summary = "temp/summary/"
os.makedirs(storage_path_summary, exist_ok=True)

# SHAP Summary Plots for Each Class
for class_idx, class_name in enumerate(iris.target_names):
    print(f"SHAP Summary Plot for Class {class_name}:")
    shap.summary_plot(shap_values[..., class_idx], X_test, feature_names=iris.feature_names, show=False)
    plt.title(f"SHAP Summary Plot for Class: {class_name}")
    plt.savefig(os.path.join(storage_path_summary, f"shap_summary_plot_class_{class_name}.png"))
    plt.close()

## Shap waterfall plot for each test sample
# Create the temp directory if it doesn't exist
storage_path_waterfall = "temp/waterfall/"
os.makedirs(storage_path_waterfall, exist_ok=True)

re_index_x_test = X_test.reset_index(drop=True)
print(re_index_x_test.iloc[0, :])

# Plot SHAP Waterfall plots for each instance in X_test
for i in range(X_test.shape[0]):
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(values=shap_values[i][:, 0],  # SHAP values for the 2nd class (index 1)
                         base_values=explainer.expected_value[0],
                         data=X_test.iloc[i],
                         feature_names=X_train.columns),
        show=False  # Disable immediate display
    )
    # Save plot to file
    plot_path = os.path.join(storage_path_waterfall, f"shap_waterfall_instance_{i}.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)  # Close the figure to avoid display
    print(f"Saved SHAP Waterfall plot for instance {i} to {plot_path}")