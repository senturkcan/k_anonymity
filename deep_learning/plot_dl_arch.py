import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np


def plot_neural_network_architecture(input_dim):
    """
    Create a visual representation of the neural network architecture
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Define the architecture
    layers = [
        {'name': 'Input Layer', 'neurons': input_dim, 'type': 'input'},
        {'name': 'Dense (128)', 'neurons': 128, 'type': 'dense'},
        {'name': 'BatchNorm', 'neurons': 128, 'type': 'batchnorm'},
        {'name': 'Dropout (0.3)', 'neurons': 128, 'type': 'dropout'},
        {'name': 'Dense (64)', 'neurons': 64, 'type': 'dense'},
        {'name': 'BatchNorm', 'neurons': 64, 'type': 'batchnorm'},
        {'name': 'Dropout (0.2)', 'neurons': 64, 'type': 'dropout'},
        {'name': 'Dense (32)', 'neurons': 32, 'type': 'dense'},
        {'name': 'BatchNorm', 'neurons': 32, 'type': 'batchnorm'},
        {'name': 'Dense (16)', 'neurons': 16, 'type': 'dense'},
        {'name': 'BatchNorm', 'neurons': 16, 'type': 'batchnorm'},
        {'name': 'Output (5)', 'neurons': 5, 'type': 'output'}
    ]

    # Color scheme for different layer types
    colors = {
        'input': '#E8F4FD',
        'dense': '#4A90E2',
        'batchnorm': '#F5A623',
        'dropout': '#D0021B',
        'output': '#7ED321'
    }

    # Position parameters
    layer_width = 1.2
    layer_spacing = 1.5
    max_height = 8

    # Calculate positions
    total_width = len(layers) * layer_spacing
    start_x = -total_width / 2

    # Draw layers
    for i, layer in enumerate(layers):
        x = start_x + i * layer_spacing

        # Calculate height based on number of neurons (with scaling)
        if layer['neurons'] > 100:
            height = max_height * 0.8
        elif layer['neurons'] > 50:
            height = max_height * 0.6
        elif layer['neurons'] > 20:
            height = max_height * 0.4
        else:
            height = max_height * 0.3

        y = -height / 2

        # Draw layer box
        if layer['type'] == 'dropout':
            # Draw dropout as dashed rectangle
            rect = patches.Rectangle((x - layer_width / 2, y), layer_width, height,
                                     linewidth=2, edgecolor=colors[layer['type']],
                                     facecolor='white', linestyle='--', alpha=0.7)
        else:
            rect = FancyBboxPatch((x - layer_width / 2, y), layer_width, height,
                                  boxstyle="round,pad=0.05",
                                  facecolor=colors[layer['type']],
                                  edgecolor='black', linewidth=1.5, alpha=0.8)

        ax.add_patch(rect)

        # Add layer name
        ax.text(x, y + height + 0.3, layer['name'], ha='center', va='bottom',
                fontsize=10, fontweight='bold')

        # Add neuron count
        ax.text(x, y + height / 2, f"{layer['neurons']}\nneurons",
                ha='center', va='center', fontsize=9,
                color='white' if layer['type'] in ['dense', 'output'] else 'black')

        # Draw connections to next layer
        if i < len(layers) - 1:
            next_x = start_x + (i + 1) * layer_spacing
            # Draw multiple connection lines
            for j in range(3):  # Draw 3 connection lines as representation
                y_start = y + height * (0.2 + j * 0.3)
                y_end = y + height * (0.2 + j * 0.3)  # Same height for simplicity
                ax.plot([x + layer_width / 2, next_x - layer_width / 2],
                        [y_start, y_end], 'k-', alpha=0.3, linewidth=0.8)

    # Add activation functions annotations
    activation_positions = [1, 4, 7, 9]  # Positions of dense layers with ReLU
    for pos in activation_positions:
        x = start_x + pos * layer_spacing
        ax.text(x, max_height / 2 + 1, 'ReLU', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7),
                fontsize=8)

    # Add softmax for output
    output_x = start_x + (len(layers) - 1) * layer_spacing
    ax.text(output_x, max_height / 2 + 1, 'Softmax', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7),
            fontsize=8)

    # Set plot properties
    ax.set_xlim(start_x - 1, start_x + total_width + 1)
    ax.set_ylim(-max_height / 2 - 2, max_height / 2 + 3)
    ax.set_aspect('equal')
    ax.axis('off')

    # Add title
    #plt.title('Neural Network Architecture for Health Status Prediction',
    #          fontsize=16, fontweight='bold', pad=20)

    # Add legend
    legend_elements = [
        patches.Patch(color=colors['input'], label='Input Layer'),
        patches.Patch(color=colors['dense'], label='Dense Layer'),
        patches.Patch(color=colors['batchnorm'], label='Batch Normalization'),
        patches.Patch(color=colors['dropout'], label='Dropout'),
        patches.Patch(color=colors['output'], label='Output Layer')
    ]

    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    return fig


# Create a detailed model summary plot
def plot_model_summary_table():
    """
    Create a table showing detailed model architecture
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Model architecture data
    model_data = [
        ['Layer Type', 'Output Shape', 'Parameters', 'Activation'],
        ['Input', f'(None, {input_dim})', '0', '-'],
        ['Dense', '(None, 128)', f'{input_dim * 128 + 128:,}', 'ReLU'],
        ['BatchNormalization', '(None, 128)', '512', '-'],
        ['Dropout', '(None, 128)', '0', '-'],
        ['Dense', '(None, 64)', f'{128 * 64 + 64:,}', 'ReLU'],
        ['BatchNormalization', '(None, 64)', '256', '-'],
        ['Dropout', '(None, 64)', '0', '-'],
        ['Dense', '(None, 32)', f'{64 * 32 + 32:,}', 'ReLU'],
        ['BatchNormalization', '(None, 32)', '128', '-'],
        ['Dense', '(None, 16)', f'{32 * 16 + 16:,}', 'ReLU'],
        ['BatchNormalization', '(None, 16)', '64', '-'],
        ['Dense (Output)', '(None, 5)', f'{16 * 5 + 5:,}', 'Softmax']
    ]

    # Create table
    table = ax.table(cellText=model_data[1:], colLabels=model_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color the header
    for i in range(len(model_data[0])):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color alternate rows
    for i in range(1, len(model_data)):
        for j in range(len(model_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')

    ax.axis('off')
    plt.title('Neural Network Model Summary', fontsize=16, fontweight='bold', pad=20)

    return fig


# Example usage (you would replace input_dim with your actual input dimension)
# For demonstration, let's assume input_dim = 100 (you should use your actual value)
input_dim = 100  # Replace with your actual input dimension

# Create the architecture visualization
fig1 = plot_neural_network_architecture(input_dim)
plt.savefig('neural_network_architecture.png', dpi=300, bbox_inches='tight')
plt.show()

# Create the model summary table
fig2 = plot_model_summary_table()
plt.savefig('model_summary_table.png', dpi=300, bbox_inches='tight')
plt.show()

print("Neural network architecture plots saved as:")
print("- neural_network_architecture.png")
print("- model_summary_table.png")