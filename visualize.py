# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

# %%
fname = "data/walkability.csv"
val_df = pd.read_csv(fname)
val_df.head()

# %% [markdown]
# ### Accuracy across different threshold

# %%
def get_prediction(x, threshold):
    x = 1/(1+ np.exp(-x))
    return 0 if x < threshold else 1
thresholds = [0.05*i for i in range(21)]
val_df["diff(R-L)"] = val_df["prediction_right"] - val_df["prediction_left"]

tpr_ls = []
fpr_ls = []
for t in thresholds:
    val_df[f"t_{int(t*100)}"] = val_df["diff(R-L)"].map(lambda x: get_prediction(x, t))
    TP = val_df[(val_df[f"t_{int(t*100)}"] == 1) & (val_df[f"choice"] == 1)].shape[0]
    TN = val_df[(val_df[f"t_{int(t*100)}"] == 0) & (val_df[f"choice"] == 0)].shape[0]
    FP = val_df[(val_df[f"t_{int(t*100)}"] == 1) & (val_df[f"choice"] == 0)].shape[0]
    FN = val_df[(val_df[f"t_{int(t*100)}"] == 0) & (val_df[f"choice"] == 1)].shape[0]

    TPR = TP / (TP + FN)
    FPR = FP / (TN + FP)
    tpr_ls.append(TPR)
    fpr_ls.append(FPR)
    print(f"Accuracy({t:.2f}):", (val_df["choice"] == val_df[f"t_{int(t*100)}"]).astype(int).sum()/val_df.shape[0])


# %%
from sklearn.metrics import roc_auc_score
def sigmoid(x):
    return 1/(1 + np.exp(-x))
auc = roc_auc_score(val_df["choice"], val_df["diff(R-L)"].map(lambda x: sigmoid(x)))

plt.plot(fpr_ls, tpr_ls, label = "ROC Curve")
plt.scatter(fpr_ls, tpr_ls, marker='o')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label = "Random Guess")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Receiver Operating Characteristic (AUC = {auc})')
plt.legend(loc="lower right")
plt.show()

# %%
data = val_df[val_df["choice"] == 0]
plt.scatter(data["prediction_left"], data["prediction_right"], marker=".",  color="red", label="left win")
plt.xlabel("prediction_left")
data = val_df[val_df["choice"] == 1]
plt.scatter(data["prediction_left"], data["prediction_right"], marker=".",  color="blue", label="right win")

min_val = min(val_df["prediction_left"].min(), val_df["prediction_right"].min())
max_val = max(val_df["prediction_left"].max(), val_df["prediction_right"].max())

# Create a line from (min_val, min_val) to (max_val, max_val)
plt.plot([min_val, max_val], [min_val, max_val], 'k--')  # 'k--' creates a black dashed line
plt.title("Scatter Plot of the Validation Set")
plt.ylabel("prediction_right")
plt.legend()
plt.show()

# %%
data = val_df[val_df["choice"] == 0]
plt.hist(data["diff(R-L)"], bins = 50, color="red", label="left win")
data = val_df[val_df["choice"] == 1]
plt.hist(data["diff(R-L)"], bins = 50, color="blue", label="right win")
plt.title("Distribution of Prediction Difference")
plt.xlabel("diff(R-L)")
plt.legend()
plt.show()


