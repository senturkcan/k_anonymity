import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import export_text
import graphviz

# Load the dataset
file_path = 'new_dataset.xlsx'
data = pd.read_excel(file_path)


target_column = 'GENHLTH'
data = data.dropna(subset=[target_column])

features = [col for col in data.columns if col != target_column]

# Convert categorical features
for col in features:
    if data[col].dtype == 'object':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

X = data[features]
y = data[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Train the decision tree model 
model = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=100,
    random_state=42
)

model.fit(X_train, y_train)

# Visualization 1: Using matplotlib with plot_tree
plt.figure(figsize=(25, 15))
plot_tree(model,
          feature_names=features,
          class_names=[str(c) for c in model.classes_],
          filled=True,
          rounded=True,
          fontsize=8)
plt.title('Decision Tree Structure Visualization', fontsize=16, fontweight='bold')
plt.savefig('decision_tree_structure.png', dpi=300, bbox_inches='tight')
plt.show()
print("Decision tree visualization saved as 'decision_tree_structure.png'")

# Visualization 2: Using graphviz
try:
    dot_data = export_graphviz(model,
                               out_file=None,
                               feature_names=features,
                               class_names=[str(c) for c in model.classes_],
                               filled=True,
                               rounded=True,
                               special_characters=True)

    # Save as .dot
    with open('decision_tree.dot', 'w') as f:
        f.write(dot_data)
    print("Decision tree DOT file saved as 'decision_tree.dot'")

    # create .pdf/.png
    try:
        graph = graphviz.Source(dot_data)
        graph.render('decision_tree_graphviz', format='png', cleanup=True)
        print("High-quality decision tree visualization saved as 'decision_tree_graphviz.png'")
    except:
        print("Graphviz not available for rendering. DOT file created instead.")
        print("You can visualize the DOT file online at: http://magjac.com/graphviz-visual-editor/")

except Exception as e:
    print(f"Graphviz export failed: {e}")


# Visualization 3: Create a simplified tree structure plot
def plot_simplified_tree(model, features, max_depth_show=4):
    """Create a simplified tree visualization showing only the top levels"""

    # Create a new model with limited depth for visualization
    simple_model = DecisionTreeClassifier(
        max_depth=max_depth_show,
        min_samples_split=100,
        random_state=42
    )
    simple_model.fit(X_train, y_train)

    plt.figure(figsize=(20, 12))
    plot_tree(simple_model,
              feature_names=features,
              class_names=[str(c) for c in simple_model.classes_],
              filled=True,
              rounded=True,
              fontsize=10,
              impurity=True,
              proportion=True)

    plt.title(f'Simplified Decision Tree (Top {max_depth_show} levels)',
              fontsize=16, fontweight='bold')
    plt.savefig(f'decision_tree_simplified_depth_{max_depth_show}.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Simplified tree visualization saved as 'decision_tree_simplified_depth_{max_depth_show}.png'")


# Create simplified version
plot_simplified_tree(model, features, max_depth_show=4)


# Visualization 4: Text-based tree structure
def create_formatted_tree_text(model, features):
    """Create a well-formatted text representation of the tree"""
    tree_rules = export_text(model, feature_names=features, show_weights=True)

    # Save as .txt
    with open('formatted_decision_tree_rules.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DECISION TREE STRUCTURE\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model Parameters:\n")
        f.write(f"- Max Depth: {model.max_depth}\n")
        f.write(f"- Min Samples Split: {model.min_samples_split}\n")
        f.write(f"- Number of Features: {len(features)}\n")
        f.write(f"- Number of Classes: {len(model.classes_)}\n")
        f.write(f"- Tree Depth: {model.tree_.max_depth}\n")
        f.write(f"- Number of Nodes: {model.tree_.node_count}\n")
        f.write(f"- Number of Leaves: {model.tree_.n_leaves}\n\n")
        f.write("=" * 80 + "\n")
        f.write("TREE RULES:\n")
        f.write("=" * 80 + "\n\n")
        f.write(tree_rules)

    print("Formatted decision tree rules saved as 'formatted_decision_tree_rules.txt'")


create_formatted_tree_text(model, features)

# Visualization 5: Create a summary of tree statistics
print("\n" + "=" * 50)
print("DECISION TREE STATISTICS")
print("=" * 50)
print(f"Tree depth: {model.tree_.max_depth}")
print(f"Number of nodes: {model.tree_.node_count}")
print(f"Number of leaves: {model.tree_.n_leaves}")
print(f"Number of features used: {np.sum(model.feature_importances_ > 0)}")

# Show the most important decision nodes
print(f"\nTop 5 most important features in the tree:")
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]
for i in range(min(5, len(features))):
    idx = sorted_idx[i]
    print(f"{i + 1}. {features[idx]}: {feature_importance[idx]:.4f}")

print("\nMaking visualizations")