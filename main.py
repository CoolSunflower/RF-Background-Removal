import streamlit as st
import cv2
import numpy as np
from background_removal import remove_background_with_bbox
from PIL import Image
import io
import base64
import streamlit_drawable_canvas as dc

st.set_page_config(layout="wide")

st.title("Team Amber: Background Removal using Random Forests")

# Two-column layout
col1, col2 = st.columns(2)

with col1:
    st.header("Upload an Image and Draw Bounding Boxes")
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # st.image(image, caption="Original Image", use_container_width=True)
        image_np = np.array(image)

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        background_image_url = f"data:image/png;base64,{img_b64}"

        canvas_result = dc.st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=image,
            update_streamlit=True,
            height=image.size[1],
            width=image.size[0],
            drawing_mode="rect",
            display_toolbar=True,
            key="canvas",
        )

        # Add a slider for morph_kernel_size
        morph_kernel_size = st.slider(
            "Morphological Kernel Size", 
            min_value=1, 
            max_value=15, 
            value=5, 
            step=1, 
            help="Adjust the size of the morphological kernel used in background removal"
        )

        # Add a slider for max hp tuning iter
        max_iter = st.slider(
            "Hyperparameter Tuning Iterations", 
            min_value=10, 
            max_value=100, 
            value=10, 
            step=2, 
            help="Adjust the number of iterations for hyperparameter tuning of the random forest model"
        )

        # Collect bounding boxes
        bboxes = []
        if canvas_result.json_data is not None:
            for obj in canvas_result.json_data["objects"]:
                left = int(obj["left"])
                top = int(obj["top"])
                width = int(obj["width"])
                height = int(obj["height"])
                bboxes.append((left, top, left + width, top + height))

        # Button to remove background
        if st.button("Remove Background") and bboxes:
            st.session_state.iteration = 0
            result_mask = remove_background_with_bbox(image_np, bboxes, max_samples=10000, morph_kernel_size=morph_kernel_size, max_hp_tuning_iter=max_iter, features=['lbp', 'ltp', 'quest', 'hog'])
            masked_image = image_np.copy()
            masked_image[result_mask == 0] = [0, 0, 0]
            st.success("Background removed!")

            # Display the result in the second column
            with col2:
                st.header("Result")
                st.image(masked_image, caption="Image with Background Removed", use_container_width=True)
