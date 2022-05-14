# %%
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

from utils import load_accs

cocostuff_accs_file = "../evaluation/train_with_gt_bboxes/test_on_cocostuff/accuracies.txt"
unrel_accs_file = "../evaluation/train_with_gt_bboxes/test_on_unrel/accuracies.txt"
ind2label_file = "/home/philipp/Git/out-of-context-benchmark/data/COCOstuff/idx2label.txt"
plotdir = "../visualizations" # set None if you don't want to save plots

# %% load files

cocostuff_class_idxs, cocostuff_class_accs, cocostuff_total_acc = load_accs(cocostuff_accs_file)
unrel_classes_idxs, unrel_class_accs, unrel_total_acc = load_accs(unrel_accs_file)

assert all([a == b for a,b in zip(cocostuff_class_idxs, unrel_classes_idxs)]), "class indexes don't match"

# %% load index-label mapping
idx2label = {}
with open(ind2label_file) as f:
    for line in f:
        s = line.split(":")
        idx2label[int(s[0])] = s[1].strip()

# %% plot class accuracies of cocostuff vs unrel
index = np.arange(len(cocostuff_class_idxs))
labels = [idx2label[i] for i in cocostuff_class_idxs]
bar_width = 0.35

fig, ax = plt.subplots()
ax.bar(index, cocostuff_class_accs, bar_width, label="COCOstuff")
ax.bar(index + bar_width, unrel_class_accs, bar_width, label="UnRel")

#ax.set_xlabel('Category')
ax.set_ylabel('Accuracy')
#ax.set_title('Crime incidence by season, type')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(labels, rotation="vertical")
ax.legend()
plt.tight_layout()

if plotdir is not None:
    plt.savefig(plotdir + "/cocostuff_vs_unrel.png", dpi=300)
else:
    plt.show()

# %% plot differences between accuracies
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 6]})
plt.xlabel(r"$acc_{COCOstuff} - acc_{UnRel}$")
plt.xlim(-1,1)

ax1.hist(cocostuff_class_accs - unrel_class_accs)
ax1.axvline(x=0, color="black")

ax2.scatter(cocostuff_class_accs - unrel_class_accs, index)
ax2.axvline(x=0, color="black")
ax2.set_yticks(index)
ax2.set_yticklabels(labels, fontsize=7)
ax2.text(-0.98, index[-4], "Higher accuracy\non UnRel", color="red")
ax2.text( 0.5, index[-4], "Higher accuracy\non COCOstuff", color="green")

#plt.show()
plt.tight_layout()
plt.subplots_adjust(hspace=0.05)

if plotdir is not None:
    plt.savefig(plotdir + "/cocostuff_vs_unrel_differences_hist.png", dpi=300)
else:
    plt.show()