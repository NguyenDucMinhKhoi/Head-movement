import matplotlib.pyplot as plt
import numpy as np

# Example data
configurations = ['Tai Baseline', 'Config 1', 'Config 2', 'Config 3']
mae_protocol_1 = [4.08, 3.88, 13.06, 3.85]  # MAE values for Protocol 1
mae_protocol_2 = [4.58, 4.90, 13.29, 4.84]  # MAE values for Protocol 2


x = np.arange(len(configurations))  # The label locations
width = 0.35  # The width of the bars

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, mae_protocol_1, width, label='Protocol 1')
bars2 = ax.bar(x + width/2, mae_protocol_2, width, label='Protocol 2')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Configuration')
ax.set_ylabel('MAE')
ax.set_xticks(x)
ax.set_xticklabels(configurations, rotation=0, ha='center')  # Labels are horizontal
ax.legend()

# Add value labels
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_value_labels(bars1)
add_value_labels(bars2)

fig.tight_layout()

plt.show()
