"""
Architecture
"""

import csv
import matplotlib.pyplot as plt

# Data for Figure 1 (28 datasets average)
models_fig1 = ['Mamba (CLM)', 'Transformer (CLM)', 'Transformer (MLM)', 'MoE Transformer (CLM)', 'MoE Transformer (MLM)', 'DNABERT-2']
training_cost_hours_fig1 = [7, 7 + 12/60, 7 + 12/60, 20, 20, 70]  # Convert training times to hours
performance_fig1 = [67.12, 67.31, 64.22, 68.49, 64.81, 66.80]
model_sizes_fig1 = [93, 112, 112, 707, 707, 117]

# Data for Figure 2 (18 datasets average)
models_fig2 = ['Mamba (CLM)', 'Transformer (CLM)', 'Transformer (MLM)', 'MoE Transformer (CLM)', 'MoE Transformer (MLM)', 'DNABERT-2']
training_cost_hours_fig2 = [7, 7 + 12/60, 7 + 12/60, 20, 20, 70]  # Convert training times to hours
performance_fig2 = [69.31, 69.61, 68.23, 69.87, 68.35, 72.82]
model_sizes_fig2 = [93, 112, 112, 707, 707, 117]

# Training loss data
loss_file = "/Users/ZZH/Documents/loss_iclr2025.csv"
with open(loss_file, "r") as f:
    reader = list(csv.reader(f))[1:]
    loss_mamba = [float(l[5]) for l in reader]
    loss_transformer_clm = [float(l[4]) for l in reader]
    loss_transformer_mlm = [float(l[2]) for l in reader]
    loss_moe_transformer_clm = [float(l[3]) for l in reader]
    loss_moe_transformer_mlm = [float(l[1]) for l in reader]
    train_steps = [int(l[0]) for l in reader]

losses = {
    'Mamba (CLM)': loss_mamba,
    'Transformer (CLM)': loss_transformer_clm,
    'Transformer (MLM)': loss_transformer_mlm,
    'MoE Transformer (CLM)': loss_moe_transformer_clm,
    'MoE Transformer (MLM)': loss_moe_transformer_mlm,
}

# Colors for both figures
colors_fig1_fig2 = ['#9b5de5', '#ef476f', '#ffd166', '#06d6a0', '#118ab2', '#073b4c']
# background_color = '#F5F5F5'
background_color = 'white'


# Font sizes
adjusted_large_font = 20
adjusted_title_font = 24
adjusted_legend_font = 14

fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor=background_color)

# Plot Figure 1 with new title
scatter1 = axes[0].scatter(training_cost_hours_fig1, performance_fig1, s=[size*5 for size in model_sizes_fig1], alpha=0.8, c=colors_fig1_fig2, marker='o', edgecolor='black')
axes[0].set_xlim(0, 75)
axes[0].set_ylim(63, 70)
axes[0].set_title('GUE results (all datasets)', fontsize=adjusted_title_font)
axes[0].set_xlabel('Training Cost (hours)', fontsize=adjusted_large_font)
axes[0].set_ylabel('Averaged Performance', fontsize=adjusted_large_font)
axes[0].set_facecolor(background_color)
axes[0].grid(True, color='gray', linestyle='--', linewidth=0.5)

# Adding legend for Figure 1 with round points
for i, model in enumerate(models_fig1):
    axes[0].scatter([], [], c=colors_fig1_fig2[i], label=model, s=100, marker='o')  # Round points for legend
axes[0].legend(title="Models", loc='lower right', fontsize=adjusted_legend_font)

# Plot Figure 2 with new title
scatter2 = axes[1].scatter(training_cost_hours_fig2, performance_fig2, s=[size*5 for size in model_sizes_fig2], alpha=0.8, c=colors_fig1_fig2, marker='o', edgecolor='black')
axes[1].set_xlim(0, 75)
axes[1].set_ylim(67, 74)
axes[1].set_title('GUE results (exclude EMP)', fontsize=adjusted_title_font)
axes[1].set_xlabel('Training Cost (hours)', fontsize=adjusted_large_font)
axes[1].set_facecolor(background_color)
axes[1].grid(True, color='gray', linestyle='--', linewidth=0.5)

# Adding legend for Figure 2 with round points
for i, model in enumerate(models_fig2):
    axes[1].scatter([], [], c=colors_fig1_fig2[i], label=model, s=100, marker='o')  # Round points for legend
axes[1].legend(title="Models", loc='lower right', fontsize=adjusted_legend_font)

# Plot the training loss in the third subplot
for i, model in enumerate(losses):
    axes[2].plot(train_steps, losses[model], label=model, color=colors_fig1_fig2[i], linewidth=2)

axes[2].set_title('Training Loss', fontsize=adjusted_title_font)
axes[2].set_xlabel('Training Steps', fontsize=adjusted_large_font)
axes[2].set_ylabel('Training Loss', fontsize=adjusted_large_font)
axes[2].set_facecolor(background_color)
axes[2].grid(True, color='gray', linestyle='--', linewidth=0.5)
axes[2].legend(loc='upper right', fontsize=adjusted_legend_font)

# Show the combined plot with updated titles
plt.tight_layout()

# Save the figure as PDF
pdf_path = 'model_performance_and_training_loss.pdf'
fig.savefig(pdf_path, format='pdf')






"""
Tokenizer part-2
"""
import matplotlib.pyplot as plt

# New data for the next plot
models_new = ['Character-level (4-dim)', 'Non-overlapping 3-mer (64-dim)', 'Non-overlapping 6-mer (4096-dim)', 'Overlapping 6-mer (4096-dim)', 'BPE (4096-dim)']
performance_new = [19.94, 39.92, 40.25, 55.41, 46.46]
compression_rate_new = [1, 3, 6, 1, 5]

# Colors for the new figure
colors_new = ['#9b5de5', '#f15bb5', '#fee440', '#00bbf9', '#06d6a0']

# Constant size for points (initial point size, later doubled)
point_size = 150
point_size_doubled = point_size * 2

# Adjusting the background color (light gray)
# new_background_color = '#f0f0f0'
new_background_color = 'white'

fig, ax = plt.subplots(figsize=(6, 6), facecolor=new_background_color)  # Reduced width by ~40%

# Create scatter plot with doubled point size
scatter = ax.scatter(compression_rate_new, performance_new, s=point_size_doubled, alpha=0.8, c=colors_new, marker='o', edgecolor='black')

# Set axis limits and labels (no title)
ax.set_xlim(0, 7)
ax.set_ylim(15, 60)
ax.set_xlabel('Compression Rate', fontsize=20)
ax.set_ylabel('Averaged Performance', fontsize=20)
ax.set_facecolor(new_background_color)
ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

# Adding legend for the models
for i, model in enumerate(models_new):
    ax.scatter([], [], c=colors_new[i], label=model, s=100, marker='o')  # Empty points for legend

ax.legend(title="Models", loc='lower right', fontsize=14)

# Show the plot with adjusted width and no title
plt.tight_layout()

# Save the figure as a PDF
pdf_path_adjusted_width = 'preliminary_tokenizer_b.pdf'
fig.savefig(pdf_path_adjusted_width, format='pdf')



"""
Embedding Efficiency
"""
import matplotlib.pyplot as plt

# Data for plotting
sequence_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]
genomeocean_times = [1.002712965, 1.733072758, 3.1565516, 6.460528851, 12.76856041, 24.17451525, 54.24956989]
genslm_times = [8.405895948, 13.2921226, 27.97950101, 62.72054672, None, None, None]
evo_times = [17.53504014, 34.25295091, 66.73957229, None, None, None, None]

# Custom colors for each model
colors = ['#ffd166', '#00bbf9', '#06d6a0']

# Set up the plot
plt.figure(figsize=(10, 6))

# Plot each model with larger points and wider lines, using "o" marker for all
plt.plot(sequence_lengths, genomeocean_times, label='GenomeOcean', marker='o', color=colors[0], markersize=10, linewidth=3.5)
plt.plot(sequence_lengths[:4], genslm_times[:4], label='GenSLMs', marker='o', color=colors[1], markersize=10, linewidth=3.5)
plt.plot(sequence_lengths[:3], evo_times[:3], label='Evo', marker='o', color=colors[2], markersize=10, linewidth=3.5)

# Labels and Title
plt.xlabel('Sequence Length (bp)')
plt.ylabel('Time (seconds)')
plt.title('Time Comparison of Models for Forward Pass on A100 80G GPU')

# Add legend
plt.legend()

# Grid and layout adjustments
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
plt.show()





"""
Similarity Distribution - Context
"""

import json
import numpy as np
import matplotlib.pyplot as plt

# Simulate data: 5 lists of 1000 values each, ranging from 0 to 1
np.random.seed(42)
with open("/Users/ZZH/Downloads/similarity/real_evo.json", "r") as f:
    real_evo = json.load(f)
with open("/Users/ZZH/Downloads/similarity/real_go.json", "r") as f:
    real_go = json.load(f)
with open("/Users/ZZH/Downloads/similarity/real_genslm.json", "r") as f:
    real_genslm = json.load(f)
with open("/Users/ZZH/Downloads/similarity/real_reorder.json", "r") as f:
    real_reorder = json.load(f)
with open("/Users/ZZH/Downloads/similarity/real_real.json", "r") as f:
    real_real = json.load(f)
    
data = [real_real, real_go, real_evo, real_genslm, real_reorder]
data = [np.array(d) for d in data]

# Define specific colors for each violin
colors = ['#9b5de5', '#118ab2', '#ffd166', '#06d6a0', '#ef476f']

# Create a violin plot
plt.figure(figsize=(10, 7))
parts = plt.violinplot(data, showmeans=False, showmedians=True)

# Set specific colors for each violin
for i in range(len(data)):
    parts['bodies'][i].set_facecolor(colors[i])
    parts['bodies'][i].set_edgecolor(colors[i])
    parts['bodies'][i].set_alpha(0.8)

# Customize the median line color to black
for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
    vp = parts[partname]
    vp.set_edgecolor('black')  # Make the lines black
    vp.set_linewidth(1.5)

# Add gridlines, and customize background
plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
plt.gca().set_facecolor('#f7f7f7')

# Add a box around the plot
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_color('#333333')
plt.gca().spines['bottom'].set_color('#333333')

# Add title, labels, and customize font
# plt.title("Violin Plot of Genome Models' Similarity", fontsize=20, fontweight='bold', color='#333333', pad=20)
plt.xlabel("")  # Remove x label
plt.ylabel("Value Distribution (0 to 1)", fontsize=24, fontweight='bold', color='#333333')
plt.xticks(ticks=[1, 2, 3, 4, 5], labels=["Real", "GenomeOcean", "Evo", "GenSLM", "Reorder"], fontsize=22, fontweight='bold')
plt.yticks(fontsize=20, color='#333333')

# Display the plot
plt.tight_layout()
# plt.show()
plt.savefig('/Users/ZZH/Downloads/similarity.pdf', format='pdf')


















"""
Similarity Distribution - Ground Truth
"""








































# '#f15bb5', '#fee440', '#00bbf9'

"""
Generation Efficiency
"""

import matplotlib.pyplot as plt

adjusted_large_font = 20
adjusted_title_font = 24
adjusted_legend_font = 14

# Data for the models and their throughput
models_separate = ['GenomeOcean ', 'GenomeOcean', 'Evo ', 'Evo', 'GenSLMs']
throughput_separate = [12487.8, 3938.461538, 81.92, 82.78092158, 157.5384615]

# Colors for the models
colors = ['#ffd670', '#ffd670', '#06d6a0', '#06d6a0', '#118ab2']

# Creating a bar plot with enhanced styling and bold model names
plt.figure(figsize=(9, 6))

# Bar plot with five separate bars
bars = plt.bar(models_separate, throughput_separate, color=colors, edgecolor='black', linewidth=1.2, log=True)

# Adding a subtle shadow to the bars for 3D effect
for bar in bars:
    bar.set_zorder(2)

# Adjusting the position of the text labels to move them down further
# for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2, yval * 0.5, round(yval, 1), ha='center', va='bottom', color='black', fontsize=16, fontweight='bold', zorder=3)

# Removing xlabel
plt.xlabel('')

# Making x-tick labels bold
plt.xticks('')

# Logarithmic scale for y-axis and setting y-axis limit to close to 10**5
# plt.yscale('log', base=2)
# plt.ylim(50, 2*10**4)
plt.yticks(fontsize=16, fontweight='bold')

# Remove the top and right spines (boundary)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

# Background adjustment and title
plt.gca().set_facecolor('white')  
plt.ylabel('Throughput (Base/Second)', fontsize=20, fontweight='bold')
# plt.title('Throughput Comparison of Generative Models (Bolded Labels)', fontsize=adjusted_title_font, fontweight='bold')

# Show the plot
plt.tight_layout()
# plt.savefig('/Users/ZZH/Downloads/generation_efficiency.pdf', format='pdf')
plt.show()







"""
Impact of context length
"""

import matplotlib.pyplot as plt
import numpy as np

# Data for the two groups
group1 = [
    [50.85, 61.03, 68.11, 70.44, 73.22, 72.50, 90.03],  # DNABERT-2
    [48.81, 58.10, 66.63, 69.24, 73.38, 73.17, 88.61],  # NT-v2
    [46.00, 55.95, 62.81, 66.23, 71.10, 68.66, 86.01],  # HyenaDNA
    [44.89, 54.42, 62.87, 64.14, 66.07, 64.71, 81.30]   # Caduceus
]

group2 = [
    [73.38, 80.53, 81.46, 81.70, 82.66, 83.36, 90.03],  # DNABERT-2
    [68.63, 76.92, 79.13, 78.48, 79.65, 80.78, 88.61],  # NT-v2
    [57.07, 70.76, 73.73, 75.75, 77.70, 77.73, 86.01],  # HyenaDNA
    [44.80, 57.09, 64.22, 65.90, 67.61, 71.24, 81.30]   # Caduceus
]

labels = ["DNABERT-2", "NT-v2", "HyenaDNA", "Caduceus"]
x_labels = ["0.5k", "1k", "2k", "4k", "8k", "16k", "Real"]

# Create subplots side by side with updated titles and no background lines
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 2))

# Plot the first group with new title
cax1 = ax1.imshow(group1, cmap="viridis", aspect="auto", vmin=30, vmax=90)
fig.colorbar(cax1, ax=ax1)
ax1.set_xticks(np.arange(len(x_labels)))
ax1.set_yticks(np.arange(len(labels)))
ax1.set_xticklabels(x_labels, fontweight='bold')  # Bold x-axis labels
ax1.set_yticklabels(labels, fontweight='bold')
ax1.set_title('Real (Train) -> Generated (Val/Test)', fontweight='bold')

# Plot the second group with new title
cax2 = ax2.imshow(group2, cmap="viridis", aspect="auto", vmin=30, vmax=90)
fig.colorbar(cax2, ax=ax2)
ax2.set_xticks(np.arange(len(x_labels)))
ax2.set_yticks(np.arange(len(labels)))
ax2.set_xticklabels(x_labels, fontweight='bold')  # Bold x-axis labels
ax2.set_yticklabels(labels, fontweight='bold')
ax2.set_title('Generated (Train) -> Real (Val/Test)', fontweight='bold')

# Add text annotations inside the boxes for both groups
for i in range(len(group1)):
    for j in range(len(group1[i])):
        ax1.text(j, i, f"{group1[i][j]:.2f}", ha="center", va="center", color="black")
        ax2.text(j, i, f"{group2[i][j]:.2f}", ha="center", va="center", color="black")

# Remove background grid and lines
ax1.grid(False)
ax2.grid(False)

plt.tight_layout()
# plt.show()
plt.savefig('/Users/ZZH/Downloads/context_length.pdf', format='pdf')






