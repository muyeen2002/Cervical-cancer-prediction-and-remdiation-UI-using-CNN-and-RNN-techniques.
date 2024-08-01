import streamlit as st
from keras.preprocessing import image
import numpy as np
from PIL import Image
from keras.models import load_model

# Load the model (replace 'your_model_path' with the actual path to your model file)
model = load_model('cerival_cnn.h5')

# Map class indices to class names
class_names = ["Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "Superficial-Intermediate"]

# Streamlit UI
st.title("Cervical cancer prediction")

# Upload image through Streamlit UI
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    img = Image.open(uploaded_file)
    img = img.resize((75, 75))  # Adjust the target size to match your model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values to the range [0, 1]

    # Ensure the input shape matches the expected input size of the model
    if img_array.shape[1:] != (75, 75, 3):
        st.error(f"Expected image data of shape (1, 75, 75, 3), but got {img_array.shape}")
    else:
        # Display the uploaded image
        st.image(img, caption="Uploaded Image", width=200)

        # Make predictions using the loaded model
        predictions = model.predict(img_array)

        # Get the predicted class index and class name
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = class_names[predicted_class_index]

        # Display predictions
        st.write("Predicted Class Name:", predicted_class_name)
        if predicted_class_name == "Dyskeratotic":
            st.write("Dyskeratotic cells are characterized by a high nuclear-to-cytoplasmic ratio, hyperchromasia, and irregular nuclear contours.")
            st.markdown("<p style='color: red;'>Remedies:</p>", unsafe_allow_html=True)
            st.write("Dyskeratotic cells might be associated with skin issues. Treatment options could include topical medications prescribed by a dermatologist. Maintaining good skin hygiene, using moisturizers, and protecting the skin from excessive sun exposure are general suggestions.")
        elif predicted_class_name == "Koilocytotic":
            st.write("Koilocytotic cells are characterized by perinuclear halos and pyknotic nuclei.")
            st.markdown("<p style='color: red;'>Remedies:</p>", unsafe_allow_html=True)
            st.write("Koilocytotic cells may be linked to human papillomavirus (HPV) infection, which can lead to genital warts. Consultation with a healthcare professional is crucial. Treatment may involve topical medications, cryotherapy, or other medical procedures. Vaccination against certain strains of HPV is also a preventive measure.")
        elif predicted_class_name == "Metaplastic":
            st.write("Metaplastic cells are characterized by a high nuclear-to-cytoplasmic ratio and a polygonal shape.")
            st.markdown("<p style='color: red;'>Remedies:</p>", unsafe_allow_html=True)
            st.write("Metaplastic cells may indicate changes in tissue structure. Treatment depends on the underlying cause. For cervical cells, close monitoring or medical interventions might be recommended. Lifestyle factors such as avoiding tobacco and practicing safe sex may contribute to overall health.")
        elif predicted_class_name == "Parabasal":
            st.write("Parabasal cells are characterized by a high nuclear-to-cytoplasmic ratio and a round shape.")
            st.markdown("<p style='color: red;'>Remedies:</p>", unsafe_allow_html=True)
            st.write("Parabasal cells can be associated with hormonal changes or infections. Treatment may involve addressing the underlying cause. Maintaining good genital hygiene, using barrier methods during sexual activity, and seeking medical advice for any unusual symptoms are important.")
        elif predicted_class_name == "Superficial-Intermediate":
            st.write("Superficial-Intermediate cells are characterized by a low nuclear-to-cytoplasmic ratio and a round shape.")
            st.markdown("<p style='color: red;'>Remedies:</p>", unsafe_allow_html=True)
            st.write("Superficial-intermediate cells in the context of skin cells may indicate normal or transitional stages. General skincare practices, including proper cleansing, moisturizing, and sun protection, can contribute to skin health. If there are specific skin conditions, a dermatologist's consultation is advisable.")
        else:
            pass
