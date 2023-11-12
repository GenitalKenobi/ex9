import onnxruntime as ort
import numpy as np

# Create an ONNX Runtime session with the specified providers
providers = ['CPUExecutionProvider']
ort_session = ort.InferenceSession("iris_model.onnx", providers=providers)

input_data = np.array([[5.1, 3.5, 1.4, 0.2], 
                       [6.3, 2.8, 5.1, 1.5]], dtype=np.float32)

# Run inference using ONNX Runtime
predictions = ort_session.run(None, {"input": input_data})

print("Predictions:", predictions)
class_labels = ["Iris-Setosa", "Iris-Versicolor", "Iris-Virginica"]
predicted_labels = [class_labels[max(prediction, key=lambda k : prediction[k])] for prediction in predictions[1]]

print("Predicted Class Labels:", predicted_labels)
