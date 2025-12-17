import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import json
import time


IMAGE_SIZE = (299, 299)
MODEL_PATH = r"C:\Users\mohma\Desktop\Project AI\Face class\Shouk_version\Shouk_model_V1_c50_s50_ac72.h5"
MAPPING_PATH = r"C:\Users\mohma\Desktop\Project AI\Face class\Shouk_version\class_mapping.json"


@st.cache_resource
def load_face_model():
    return load_model(MODEL_PATH)

model = load_face_model()

with open(MAPPING_PATH, "r") as f:
    idx_to_label = json.load(f)
def overlay_heatmap(img, heatmap, alpha=0.28, colormap=cv2.COLORMAP_JET):

    heatmap = cv2.resize(
        heatmap,
        (img.shape[1], img.shape[0]),
        interpolation=cv2.INTER_CUBIC
    )

    heatmap = np.clip(heatmap, 0, 1)
    heatmap = np.uint8(255 * heatmap)

    
    heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)

    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)

    heatmap_color = cv2.applyColorMap(heatmap, colormap)

    overlay = cv2.addWeighted(img, 0.74, heatmap_color, alpha, 0)

    return overlay



def predict_top3(frame):
    img = cv2.resize(frame, IMAGE_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]
    top3_idx = preds.argsort()[-3:][::-1]
    results = [(idx_to_label[str(i)], float(preds[i])) for i in top3_idx]

    return results, img, preds


def make_gradcam_heatmap(img_array, model, class_index, last_conv_layer_name=None):

    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        if isinstance(predictions, list):
            predictions = predictions[0]

        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    pooled_grads = tf.nn.softmax(pooled_grads)
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    heatmap = cv2.resize(heatmap.numpy(), (299, 299), interpolation=cv2.INTER_CUBIC)
    return heatmap


def draw_label(frame, text):
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    return frame


st.title("📸 Real-time Face Recognition + Grad-CAM")
st.write("✅ الكاميرا تعمل الآن، وستظهر أعلى 3 توقعات لكل شخص مكتشف مع Grad-CAM.")

run = st.checkbox("تشغيل الكاميرا")
show_gradcam = st.checkbox("تفعيل Grad-CAM Overlay")

frame_area = st.empty()
top3_area = st.empty()

cap = cv2.VideoCapture(0)

last_update = time.time()
last_top3 = [("Unknown", 0.0), ("Unknown", 0.0), ("Unknown", 0.0)]
last_img_array = None
last_preds = None

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("❌ الكاميرا غير متصلة")
        break

    frame = cv2.flip(frame, 1)

    current_time = time.time()  
    if current_time - last_update >= 0.0:
        top3, img_array, preds = predict_top3(frame)
        last_top3 = top3
        last_img_array = img_array
        last_preds = preds
        last_update = current_time
    else:
        top3 = last_top3
        img_array = last_img_array
        preds = last_preds

    if show_gradcam and top3[0][0] != "Unknown":
        class_index = np.argmax(preds)
        heatmap = make_gradcam_heatmap(
                    img_array, model, class_index,
                    last_conv_layer_name="mixed10"
                )
        frame_with_cam = overlay_heatmap(frame.copy(), heatmap)
    else:
        frame_with_cam = frame.copy()

    result_img = draw_label(frame_with_cam, f"{top3[0][0]} ({top3[0][1]*100:.2f}%)")

    frame_area.image(result_img[:, :, ::-1], channels="RGB")

    with top3_area.container():
        st.subheader("🏆 Top 3 Predictions")
        for label, acc in top3:
            st.write(f"**{label}** — {acc*100:.2f}%")

cap.release()
