import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
with_mask = np.load('with_mask.npy')
without_mask = np.load('without_mask.npy')
with_mask = with_mask.reshape(200, 50*50*3)
without_mask = without_mask.reshape(200, 50*50*3)
X = np.r_[with_mask, without_mask]
labels = np.zeros(X.shape[0])
labels[200:] = 1.0
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X_scaled, labels, test_size=0.25, random_state=42)
pca = PCA(n_components=3)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)
svm = SVC(kernel='linear')
svm.fit(x_train_pca, y_train)
y_pred = svm.predict(x_test_pca)
print("Accuracy:", accuracy_score(y_test, y_pred))
joblib.dump(svm, 'svm_model.pkl')
joblib.dump(pca, 'pca_transform.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump({0: 'Mask', 1: 'No Mask'}, 'names.pkl')
