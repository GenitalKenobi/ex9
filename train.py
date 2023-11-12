from sklearn.datasets import load_iris
from sklearn.svm import NuSVC
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

df = load_iris()
X = df.data
y = df.target

model = NuSVC()
model.fit(X, y)

initial_types = [('input', FloatTensorType([None, X.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_types)

onnx.save_model(onnx_model, 'model.onnx')