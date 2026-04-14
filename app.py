import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Load model
model = YOLO("best.pt")

st.title("Aerial Object Detection (Bird vs Drone) - By PRABHASH")

# Upload image
file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Run detection
    results = model(img)

    # Show result image
    result_img = results[0].plot()
    st.image(result_img, caption="Detection Result")

    # Show labels
    st.subheader("Predictions:")
    for box in results[0].boxes:
        label = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        if conf > 0.5: 
            st.write(f"{label} - {conf*100:.0f}%")