"""
ORF
"""

import os
import csv
import numpy as np
from scipy import stats
import sklearn.metrics
from scipy.special import kl_div
import matplotlib.pyplot as plt



with open("/Users/ZZH/Downloads/genomeocean/CODE_NON_CODE/orf_stick/coding_evo_genlsm_genomeocean_ground.csv", "r") as f:
    lines_coding_raw = list(csv.reader(f))[1:]
    lines_coding = []
    for line in lines_coding_raw:
        if line[4] == '':
            line[4] = 0
        else:
            line[4] = int(line[4])
        lines_coding.append(line[4] / 2000.)
    
    
with open("/Users/ZZH/Downloads/genomeocean/CODE_NON_CODE/orf_stick/non_coding_evo_genlsm_genomeocean_ground.csv", "r") as f:
    lines_non_coding_raw = list(csv.reader(f))[1:]
    lines_non_coding = []
    for line in lines_non_coding_raw:
        if line[4] == '':
            line[4] = 0
        else:
            line[4] = int(line[4])
        lines_non_coding.append(line[4] / 2000.)
    

lines_evo = lines_coding[:500] + lines_non_coding[:500]
lines_genlsm = lines_coding[500:1000] + lines_non_coding[500:1000]
lines_genomeocean = lines_coding[1000:1500] + lines_non_coding[1000:1500]
lines_real = lines_coding[1500:] + lines_non_coding[1500:]

corr_evo, _ = stats.pearsonr(lines_evo, lines_real)
corr_genlsm, _ = stats.pearsonr(lines_genlsm, lines_real)
corr_genomeocean, _ = stats.pearsonr(lines_genomeocean, lines_real)

sp_corr_evo, _ = stats.spearmanr(lines_evo, lines_real)
sp_corr_genlsm, _ = stats.spearmanr(lines_genlsm, lines_real)
sp_corr_genomeocean, _ = stats.spearmanr(lines_genomeocean, lines_real)

mutual_info_evo = sklearn.metrics.mutual_info_score(lines_evo, lines_real)
mutual_info_genlsm = sklearn.metrics.mutual_info_score(lines_genlsm, lines_real)
mutual_info_genomeocean = sklearn.metrics.mutual_info_score(lines_genomeocean, lines_real)

kld_evo = np.mean(kl_div(lines_evo, lines_real))
kld_genlsm = np.mean(kl_div(lines_genlsm, lines_real))
kld_genomeocean = np.mean(kl_div(lines_genomeocean, lines_real))

print(f"Correlation: Evo {corr_evo}, GenSLM {corr_genlsm}, GenomeOcean {corr_genomeocean}")
print(f"Spearman Correlation: Evo {sp_corr_evo}, GenSLM {sp_corr_genlsm}, GenomeOcean {sp_corr_genomeocean}")
print(f"Mutual Information: Evo {mutual_info_evo}, GenSLM {mutual_info_genlsm}, GenomeOcean {mutual_info_genomeocean}")
print(f"KLD: Evo {kld_evo}, GenSLM {kld_genlsm}, GenomeOcean {kld_genomeocean}")



title = {
    "evo": "Evo",
    "genslm": "GenSLMs",
    "go": "GenomeOcean"
}
data = {
    "evo": lines_evo,
    "genslm": lines_genlsm,
    "go": lines_genomeocean
}
for predict in ["evo", "genslm", "go"]:
    plt.figure(figsize=(10, 10))
    plt.scatter(lines_real, data[predict], alpha=0.6, c='blue', edgecolor='k')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f"Predicted vs Actual Values ({title[predict]})")
    plt.grid(True)
    plt.savefig(f"/Users/ZZH/Downloads/Predicted_vs_Actual_{predict}.png")

"""
CAI
"""

import os
import csv
import numpy as np
from scipy import stats
import sklearn.metrics
from scipy.special import kl_div
import matplotlib.pyplot as plt


with open("/Users/ZZH/Downloads/genomeocean/CAI/cai_we_gt_evo_genlsm/Synechocystis.csv", "r") as f:
    lines_s = list(csv.reader(f))[1:]
    lines_s = [float(line[0]) for line in lines_s]
    
with open("/Users/ZZH/Downloads/genomeocean/CAI/cai_we_gt_evo_genlsm/Rhodopseudomonas_palustris.csv", "r") as f:
    lines_r = list(csv.reader(f))[1:]
    lines_r = [float(line[0]) for line in lines_r]
    
with open("/Users/ZZH/Downloads/genomeocean/CAI/cai_we_gt_evo_genlsm/Pseudomonas_aeruginosa.csv", "r") as f:
    lines_p = list(csv.reader(f))[1:]
    lines_p = [float(line[0]) for line in lines_p]
    
with open("/Users/ZZH/Downloads/genomeocean/CAI/cai_we_gt_evo_genlsm/Escherichia_coli.csv", "r") as f:
    lines_e = list(csv.reader(f))[1:]
    lines_e = [float(line[0]) for line in lines_e]
    
with open("/Users/ZZH/Downloads/genomeocean/CAI/cai_we_gt_evo_genlsm/Mycobacterium_smegmatis.csv", "r") as f:
    lines_m = list(csv.reader(f))[1:]
    lines_m = [float(line[0]) for line in lines_m]
    
with open("/Users/ZZH/Downloads/genomeocean/CAI/cai_we_gt_evo_genlsm/Bacillus_subtilis.csv", "r") as f:
    lines_b = list(csv.reader(f))[1:]
    lines_b = [float(line[0]) for line in lines_b]
    

lines_genomeocean = lines_s[:100] + lines_r[:100] + lines_p[:100] + lines_e[:100] + lines_m[:100] + lines_b[:100]
lines_real = lines_s[100:200] + lines_r[100:200] + lines_p[100:200] + lines_e[100:200] + lines_m[100:200] + lines_b[100:200]
lines_evo = lines_s[200:300] + lines_r[200:300] + lines_p[200:300] + lines_e[200:300] + lines_m[200:300] + lines_b[200:300]
lines_genlsm = lines_s[300:400] + lines_r[300:400] + lines_p[300:400] + lines_e[300:400] + lines_m[300:400] + lines_b[300:400]

corr_evo, _ = stats.pearsonr(lines_evo, lines_real)
corr_genlsm, _ = stats.pearsonr(lines_genlsm, lines_real)
corr_genomeocean, _ = stats.pearsonr(lines_genomeocean, lines_real)

sp_corr_evo, _ = stats.spearmanr(lines_evo, lines_real)
sp_corr_genlsm, _ = stats.spearmanr(lines_genlsm, lines_real)
sp_corr_genomeocean, _ = stats.spearmanr(lines_genomeocean, lines_real)

mutual_info_evo = sklearn.metrics.mutual_info_score(lines_evo, lines_real)
mutual_info_genlsm = sklearn.metrics.mutual_info_score(lines_genlsm, lines_real)
mutual_info_genomeocean = sklearn.metrics.mutual_info_score(lines_genomeocean, lines_real)

kld_evo = np.mean(kl_div(lines_evo, lines_real))
kld_genlsm = np.mean(kl_div(lines_genlsm, lines_real))
kld_genomeocean = np.mean(kl_div(lines_genomeocean, lines_real))

small_diff_evo = sum([1 for i in range(len(lines_evo)) if abs(lines_evo[i] - lines_real[i]) < 0.1]) / len(lines_evo)
small_diff_genlsm = sum([1 for i in range(len(lines_genlsm)) if abs(lines_genlsm[i] - lines_real[i]) < 0.1]) / len(lines_genlsm)
small_diff_genomeocean = sum([1 for i in range(len(lines_genomeocean)) if abs(lines_genomeocean[i] - lines_real[i]) < 0.1]) / len(lines_genomeocean)
print(f"Small Difference: Evo {small_diff_evo}, GenSLM {small_diff_genlsm}, GenomeOcean {small_diff_genomeocean}")

print(f"Correlation: Evo {corr_evo}, GenSLM {corr_genlsm}, GenomeOcean {corr_genomeocean}")
print(f"Spearman Correlation: Evo {sp_corr_evo}, GenSLM {sp_corr_genlsm}, GenomeOcean {sp_corr_genomeocean}")
print(f"Mutual Information: Evo {mutual_info_evo}, GenSLM {mutual_info_genlsm}, GenomeOcean {mutual_info_genomeocean}")
print(f"KLD: Evo {kld_evo}, GenSLM {kld_genlsm}, GenomeOcean {kld_genomeocean}")



title = {
    "evo": "Evo",
    "genslm": "GenSLMs",
    "go": "GenomeOcean"
}
data = {
    "evo": lines_evo,
    "genslm": lines_genlsm,
    "go": lines_genomeocean
}
for predict in ["evo", "genslm", "go"]:
    tolerance = 0.2  # You can adjust this value to change the band width

    plt.figure(figsize=(10, 10))
    plt.scatter(lines_real, data[predict], alpha=0.6, c='blue', edgecolor='k')
    # Add diagonal line (Perfect prediction line for [0, 1] range)
    plt.plot([0.2, 0.8], [0.2, 0.8], 'r--', label='Perfect Prediction')

    # Add a constant band around the diagonal
    plt.fill_between([0.2, 0.8], 
                    [0, 0.6], 
                    [0.4, 1], 
                    color='gray', alpha=0.2, label=f'Tolerance Band ±{tolerance}')

    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f"Predicted vs Actual Values ({title[predict]})")
    plt.grid(True)
    plt.savefig(f"/Users/ZZH/Downloads/Predicted_vs_Actual_{predict}.png")
