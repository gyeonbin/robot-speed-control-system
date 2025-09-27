import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# 예시 데이터
y_true = [0]*100 + [1]*100
y_pred = [0]*100 + [1]*100
cm = confusion_matrix(y_true, y_pred)

# 클래스 이름
labels = ["slow(0)", "fast(1)"]

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

# 컬러바
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Count", rotation=-90, va="bottom")

# 축 설정
ax.set(
    xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=labels,
    yticklabels=labels,
    ylabel="True",
    xlabel="Predicted",
    title="Performance of Logistic Regression: Confusion Matrix"
)

# 눈금 맞추기
plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

# 빨간 글씨로 숫자 넣기
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="red", fontsize=10)

# 격자선
ax.spines[:].set_visible(False)
ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
ax.tick_params(which="minor", bottom=False, left=False)

plt.tight_layout()
plt.show()
