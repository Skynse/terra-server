import io
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import flask
import onnx
import onnxruntime as ort
# load onnx model instead to see if it works
model = onnx.load('model.onnx')

session = ort.InferenceSession('model.onnx')


input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
# print input shape
print(session.get_inputs()[0].shape)



attributes = [
    "balancing_elements",
    "color_harmony",
    "content",
    "depth_of_field",
    "light",
    "motion_blur",
    "object",
    "repetition",
    "rule_of_thirds",
    "symmetry",
    "vivid_color",
    "score"
]

app = flask.Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image)).convert("RGB")

            # Preprocess image
            preprocess = transforms.Compose([
                transforms.Resize((input_shape[2], input_shape[3])), # Assuming input shape is [batch_size, channels, height, width]
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            image = preprocess(image).unsqueeze(0)

            input_name = session.get_inputs()[0].name

            outputs = session.run(None, {input_name: image.numpy()})

            data["predictions"] = []

            for i, attribute in enumerate(attributes):
                r = {"label": attribute, "probability": float(outputs[0][i])}
                data["predictions"].append(r)

            data["success"] = True

    return flask.jsonify(data)

if __name__ == "__main__":
    app.run()
