import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import time


model = tf.keras.models.load_model('leukemia.h5')


def main():
    st.sidebar.title("Navigation 🧭")
    sel = st.sidebar.radio("Choose where you want to go:", ["About", "Prediction", "Links"])

    if sel == "About":
        st.title("About This App 📋")
        st.write("""\
            Leukemia is a type of cancer affecting the blood and bone marrow, characterized by the abnormal growth of blood cells. Early detection can significantly improve treatment outcomes and reduce complications.

            This application uses a Convolutional Neural Network (CNN) model to predict leukemia types based on images of blood cells. By uploading an image, users can get predictions on whether the cells in the image indicate leukemia, and if so, which type.
        """)

    elif sel == "Prediction":
        st.title(":red[Leukemia Detection 🔍]")
        st.write("Upload an image to detect leukemia type.")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded Image', use_column_width=True)
            img = img.resize((150, 150))
            img_array = np.array(img) / 255.0
            img_array = img_array.reshape(1, 150, 150, 3)

            pred = st.button("Predict")
            if pred:
                with st.spinner('Analyzing the image...'):
                    time.sleep(2)
                    prediction = model.predict(img_array)
                ind = prediction.argmax()
                categories = ['Pro', 'Pre', 'Benign', 'Early']
                result = categories[ind]

                if result == 'Pro' or result == 'Pre':
                    st.success(f"**The cell shows {result} leukemia type.**")
                else:
                    st.info(f"**The cell is classified as {result}.**")

    elif sel == "Links":
        st.title("Useful Links 🔗")
        st.write("""\
            - [Google Colab Notebook](https://colab.research.google.com/#fileId=https%3A//storage.googleapis.com/kaggle-colab-exported-notebooks/project-leukemia-type-detection-47a0deac-9227-4b62-8d81-c140edd4d283.ipynb%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com/20241119/auto/storage/goog4_request%26X-Goog-Date%3D20241119T034621Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D0b11ff34095301d9485f194359e2839257c35c2e48abecdd6f291ea48309f82c62cc89f64d62e026893f988e36093031e8eb9af98b616ed9b9ab71b19b4b490362d73b692e4cde443930ba14484cbb2c8dfbf546da31cdf256c484567e107cd164891d1fdc7d12b7cc5d577bc0c6403b53112385a5ac514225e1b7d8f282a5ea9d30f8a746c74cdc797de709f065db9123eb22bd2535b2da83e42e1a36d03f693b443586cad72802a1939aece95df7a261ce6950d6fd49fc1462bff41636c7c43200258657a3ae9ba0952f9d656a51a3f80652c00a2a28fc317866e091feaacef52c7f120958bc341e0c318d73dcdffc174739ed170a20d8f7eb876a05065a96)
            - [Kaggle Dataset](https://www.kaggle.com/datasets/mehradaria/leukemia)
        """)


if __name__ == "__main__":
    main()
