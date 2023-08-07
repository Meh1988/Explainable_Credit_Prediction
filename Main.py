import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
import matplotlib.pyplot as plt


# Load the dataset
file_path = 'File.xls'  # Replace with the actual file path
data = pd.read_excel(file_path)

# Separate features and target
X = data.iloc[:, 1:-1]  # Exclude first and last columns
y = data['Y']          # Last column

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree classifier (you can replace this with any other classifier)
classifier = DecisionTreeClassifier()

# Train the classifier
classifier.fit(X_train, y_train)

# Evaluate the classifier
accuracy = classifier.score(X_test, y_test)
print(f"Accuracy of the classifier: {accuracy:.2f}")

# Get feature importances
feature_importances = classifier.feature_importances_

# Sort feature importances in descending order
sorted_indices = feature_importances.argsort()[::-1]

# Get the names of the features
feature_names = X.columns

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), feature_importances[sorted_indices])
plt.xticks(range(X.shape[1]), [feature_names[i] for i in sorted_indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.tight_layout()
plt.show()




# Get feature names
feature_names = X.columns

# Get tree structure
tree = classifier.tree_

# Extract important If-Then rules for class "1" and save to a text file
with open('rules_class1.txt', 'w') as file:
    file.write("Important rules for class 1:\n")
    def get_rules(tree, feature_names, node=0):
        if tree.children_left[node] == tree.children_right[node]:  # Leaf node
            if tree.value[node][0][1] > tree.value[node][0][0]:  # Class 1 majority
                file.write(f"If {feature_names[tree.feature[node]]} <= {tree.threshold[node]:.2f}, Then Class 1\n")
        else:
            file.write(f"If {feature_names[tree.feature[node]]} <= {tree.threshold[node]:.2f},\n")
            get_rules(tree, feature_names, tree.children_left[node])
            file.write(f"Else If {feature_names[tree.feature[node]]} > {tree.threshold[node]:.2f},\n")
            get_rules(tree, feature_names, tree.children_right[node])

    get_rules(tree, feature_names)

# Extract important If-Then rules for class "0" and save to a text file
with open('rules_class0.txt', 'w') as file:
    file.write("Important rules for class 0:\n")
    def get_rules_class0(tree, feature_names, node=0):
        if tree.children_left[node] == tree.children_right[node]:  # Leaf node
            if tree.value[node][0][0] > tree.value[node][0][1]:  # Class 0 majority
                file.write(f"If {feature_names[tree.feature[node]]} <= {tree.threshold[node]:.2f}, Then Class 0\n")
        else:
            file.write(f"If {feature_names[tree.feature[node]]} <= {tree.threshold[node]:.2f},\n")
            get_rules_class0(tree, feature_names, tree.children_left[node])
            file.write(f"Else If {feature_names[tree.feature[node]]} > {tree.threshold[node]:.2f},\n")
            get_rules_class0(tree, feature_names, tree.children_right[node])

    get_rules_class0(tree, feature_names)