import time
import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F

from models.dattnet import DATTNet

# -----------------------
# Device
# -----------------------
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)

# -----------------------
# Model
# -----------------------
model = DATTNet(num_classes=1)

model.load_state_dict(
    torch.load(
        "checkpoints/best_model.pth",
        map_location=device,
    )
)

model.to(device)
model.eval()

print("Model Loaded.")

IMAGE_SIZE = 224


def predict(image):

    original = image.copy()

    h, w = original.shape[:2]

    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image.astype(np.float32) / 255.0

    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)

    start = time.time()

    with torch.no_grad():

        output = model(image)

        output = torch.sigmoid(output)

        output = (output > 0.5).float()

    inference_time = time.time() - start

    mask = output.squeeze().cpu().numpy().astype(np.uint8) * 255

    mask = cv2.resize(mask, (w, h))

    # ---------------- Overlay ----------------

    overlay = original.copy()

    red = np.zeros_like(original)

    red[:, :, 2] = 255

    overlay = np.where(mask[..., None] == 255,
                       cv2.addWeighted(original, 0.6, red, 0.4, 0),
                       original)

    # ---------------- Statistics ----------------

    area = (mask > 0).sum()

    total = mask.shape[0] * mask.shape[1]

    area_percent = area / total * 100

    info = f"""
Inference Time : {inference_time:.3f} sec

Image Size : {w} × {h}

Predicted Polyp Area : {area_percent:.2f} %
"""

    return original, mask, overlay, info


with gr.Blocks(title="DATTNet Polyp Segmentation") as demo:

    gr.Markdown(
        """
# 🩺 DATTNet Polyp Segmentation

Upload a colonoscopy image and the trained DATTNet model will automatically segment the polyp.
"""
    )

    with gr.Row():

        input_image = gr.Image(
            type="numpy",
            label="Upload Image",
        )

    submit = gr.Button("Predict", variant="primary")

    with gr.Row():

        out1 = gr.Image(label="Original")

        out2 = gr.Image(label="Predicted Mask")

    with gr.Row():

        out3 = gr.Image(label="Overlay")

        out4 = gr.Textbox(
            label="Prediction Information",
        )

    submit.click(
        fn=predict,
        inputs=input_image,
        outputs=[
            out1,
            out2,
            out3,
            out4,
        ],
    )

demo.launch()