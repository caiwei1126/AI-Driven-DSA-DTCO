import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

BASE = Path(__file__).resolve().parent

df = pd.read_csv(BASE / "cnn_predictions.csv")
y_true = df["y_true"].tolist()
y_pred = df["y_pred"].tolist()

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["perfect", "defect"])
fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)

for text in disp.text_.ravel():
    text.set_fontsize(22)
    text.set_fontweight("bold")

plt.title("Confusion matrix for CNN", fontsize=20)
plt.xlabel("Predicted label", fontsize=18)
plt.ylabel("True label", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(BASE / "confusion_matrix.png", dpi=300)
plt.show()

precision_defect = precision_score(y_true, y_pred, pos_label=1, average="binary")
recall_defect = recall_score(y_true, y_pred, pos_label=1, average="binary")
f1_defect = f1_score(y_true, y_pred, pos_label=1, average="binary")
precision_perfect = precision_score(y_true, y_pred, pos_label=0, average="binary")
recall_perfect = recall_score(y_true, y_pred, pos_label=0, average="binary")
f1_perfect = f1_score(y_true, y_pred, pos_label=0, average="binary")

print(f"Precision (Defect): {precision_defect:.4f}")
print(f"Recall (Defect): {recall_defect:.4f}")
print(f"F1 (Defect): {f1_defect:.4f}")
print(f"Precision (Perfect): {precision_perfect:.4f}")
print(f"Recall (Perfect): {recall_perfect:.4f}")
print(f"F1 (Perfect): {f1_perfect:.4f}")
