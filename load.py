import onnxruntime as ort

ort_session = ort.InferenceSession('model.onnx', providers=['CPUExecutionProvider'])

data = [[5,3,4,2]]

prediction = ort_session.run(None, {"input":data})

classes = {0:'Iris Setosa', 1:'Iris Versicolour', 2:'Iris Virginica'}

print("Prediction: ", classes[prediction[0][0]])