
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import hashlib

from PIL import Image, ImageOps
from pathlib import Path
from datetime import datetime


# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "mnist_cnn_model.keras"
INPUT_FOLDER = BASE_DIR / "input_images"


# =====================================================
# 2. CUSTOM CSS
# =====================================================

st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
}

.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    color: #4da6ff;
    padding: 15px;
}

.subtitle {
    text-align: center;
    color: #b0bec5;
    font-size: 16px;
}

.result-box {
    background-color: #192536;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    margin-top: 10px;
}

.digit-number {
    font-size: 48px;
    font-weight: bold;
    color: #4da6ff;
    overflow-wrap: anywhere;
}

.stButton button {
    width: 100%;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD CNN MODEL
# =====================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()

except Exception as error:
    st.error(f"Unable to load CNN model: {error}")
    st.stop()


# =====================================================
# 4. SESSION STATE
# =====================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "processed_images" not in st.session_state:
    st.session_state.processed_images = set()


# =====================================================
# 5. FOREGROUND DETECTION
# =====================================================

def create_foreground_mask(image):

    gray = np.array(
        ImageOps.grayscale(image)
    )

    if gray.size == 0:
        raise ValueError("The uploaded image is empty.")

    # Estimate background using image borders
    border_pixels = np.concatenate([
        gray[0, :],
        gray[-1, :],
        gray[:, 0],
        gray[:, -1]
    ])

    background = float(np.median(border_pixels))

    # Difference from background
    difference = np.abs(
        gray.astype(np.float32) - background
    )

    # Adaptive threshold
    threshold = max(
        20,
        float(np.percentile(difference, 90)) * 0.18
    )

    mask = difference > threshold

    # Remove very small foreground noise
    foreground_pixels = np.count_nonzero(mask)

    minimum_pixels = max(
        12,
        int(gray.size * 0.0003)
    )

    if foreground_pixels < minimum_pixels:
        raise ValueError(
            "No digit detected. Please upload an image "
            "containing a clearly visible handwritten digit."
        )

    ys, xs = np.where(mask)

    if (
        len(xs) == 0
        or xs.max() - xs.min() < 2
        or ys.max() - ys.min() < 2
    ):
        raise ValueError(
            "No valid digit detected. Please upload a clearer image."
        )

    return mask


# =====================================================
# 6. PREPROCESS DIGIT FOR MNIST CNN
# =====================================================

def preprocess_digit(image):

    mask = create_foreground_mask(image)

    ys, xs = np.where(mask)

    cropped = mask[
        ys.min():ys.max() + 1,
        xs.min():xs.max() + 1
    ]

    digit_image = Image.fromarray(
        (cropped * 255).astype(np.uint8)
    )

    width, height = digit_image.size

    scale = 20 / max(width, height)

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    digit_image = digit_image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    canvas = Image.new("L", (28, 28), 0)

    x_position = (28 - new_width) // 2
    y_position = (28 - new_height) // 2

    canvas.paste(
        digit_image,
        (x_position, y_position)
    )

    digit_array = np.array(
        canvas,
        dtype=np.float32
    ) / 255.0

    # Center digit by its center of mass
    total = np.sum(digit_array)

    if total > 0:

        rows, columns = np.indices((28, 28))

        center_x = np.sum(
            columns * digit_array
        ) / total

        center_y = np.sum(
            rows * digit_array
        ) / total

        shift_x = int(round(13.5 - center_x))
        shift_y = int(round(13.5 - center_y))

        digit_array = np.roll(
            digit_array,
            shift_y,
            axis=0
        )

        digit_array = np.roll(
            digit_array,
            shift_x,
            axis=1
        )

    return digit_array.reshape(1, 28, 28, 1)


# =====================================================
# 7. AUTOMATIC MULTIPLE DIGIT DETECTION
# =====================================================

def get_projection_regions(projection):

    active = projection > 0

    regions = []
    start = None

    for index, value in enumerate(active):

        if value and start is None:
            start = index

        elif not value and start is not None:

            if index - start >= 3:
                regions.append((start, index))

            start = None

    if start is not None:

        if len(active) - start >= 3:
            regions.append((start, len(active)))

    return regions


def detect_digits(image):

    mask = create_foreground_mask(image)

    row_projection = np.sum(mask, axis=1)

    row_regions = get_projection_regions(
        row_projection
    )

    digit_regions = []

    for row_start, row_end in row_regions:

        row_mask = mask[row_start:row_end, :]

        column_projection = np.sum(
            row_mask,
            axis=0
        )

        column_regions = get_projection_regions(
            column_projection
        )

        for col_start, col_end in column_regions:

            region_mask = row_mask[:, col_start:col_end]

            ys, xs = np.where(region_mask)

            if len(xs) < 12:
                continue

            width = xs.max() - xs.min() + 1
            height = ys.max() - ys.min() + 1

            if width < 4 or height < 5:
                continue

            if width * height < 25:
                continue

            digit_regions.append((
                col_start,
                row_start,
                col_end,
                row_end
            ))

    # Sort from top to bottom, then left to right
    digit_regions.sort(
        key=lambda box: (box[1], box[0])
    )

    results = []

    for x1, y1, x2, y2 in digit_regions:

        results.append(
            image.crop((x1, y1, x2, y2))
        )

    return results


# =====================================================
# 8. 5 COLUMN x 2 ROW GRID PROCESSING
# =====================================================

def detect_grid_digits(
    image,
    top_margin_percent=12,
    digit_start_percent=30
):

    """
    Processes an image containing 10 samples
    arranged in 5 columns and 2 rows.
    """

    width, height = image.size

    top_margin = int(
        height * top_margin_percent / 100
    )

    grid_image = image.crop((
        0,
        top_margin,
        width,
        height
    ))

    grid_width, grid_height = grid_image.size

    cell_width = grid_width / 5
    cell_height = grid_height / 2

    results = []

    for row in range(2):

        for column in range(5):

            x1 = int(column * cell_width)
            x2 = int((column + 1) * cell_width)

            y1 = int(row * cell_height)
            y2 = int((row + 1) * cell_height)

            cell = grid_image.crop((
                x1,
                y1,
                x2,
                y2
            ))

            # Skip label area inside each cell
            cell_width_actual, cell_height_actual = cell.size

            digit_start = int(
                cell_height_actual *
                digit_start_percent / 100
            )

            digit_crop = cell.crop((
                0,
                digit_start,
                cell_width_actual,
                cell_height_actual
            ))

            results.append(digit_crop)

    return results


# =====================================================
# 9. CNN PREDICTION
# =====================================================

def predict_digit(image):

    processed = preprocess_digit(image)

    probabilities = model.predict(
        processed,
        verbose=0
    )[0]

    predicted_digit = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_digit] * 100
    )

    return (
        predicted_digit,
        confidence,
        probabilities
    )


# =====================================================
# 10. PREDICTION HISTORY
# =====================================================

def save_prediction(
    image_name,
    region,
    digit,
    confidence,
    image_hash
):

    record_key = (
        image_hash,
        region
    )

    if record_key in st.session_state.processed_images:
        return

    st.session_state.processed_images.add(
        record_key
    )

    st.session_state.prediction_history.append({

        "Image Name": image_name,
        "Region": region,
        "Predicted Digit": digit,
        "Confidence (%)": round(confidence, 2),

        "Date & Time": datetime.now().strftime(
            "%d-%m-%Y %I:%M:%S %p"
        )
    })


# =====================================================
# 11. DISPLAY SINGLE DIGIT
# =====================================================

def display_single_result(image, image_name):

    st.image(
        image,
        caption=image_name,
        width=300
    )

    try:

        digit, confidence, probabilities = predict_digit(image)

        image_hash = hashlib.md5(
            image.tobytes()
        ).hexdigest()

        save_prediction(
            image_name,
            "Single Digit",
            digit,
            confidence,
            image_hash
        )

        st.markdown(
            f"""
            <div class="result-box">
                <p>Predicted Digit</p>
                <div class="digit-number">{digit}</div>
                <p>Confidence: {confidence:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("View all digit probabilities"):

            for number, probability in enumerate(probabilities):

                st.write(
                    f"Digit {number}: {probability * 100:.2f}%"
                )

                st.progress(float(probability))

    except ValueError as error:
        st.warning(str(error))

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# =====================================================
# 12. DISPLAY MULTIPLE DIGITS
# =====================================================

def display_multiple_results(
    image,
    image_name,
    processing_mode
):

    image_hash = hashlib.md5(
        image.tobytes()
    ).hexdigest()

    # Select segmentation method
    try:

        if processing_mode == "5 x 2 Grid (10 samples)":

            regions = detect_grid_digits(
                image,
                top_margin_percent=top_margin,
                digit_start_percent=digit_start
            )

        else:

            regions = detect_digits(image)

    except ValueError as error:

        st.error(str(error))
        return

    if not regions:

        st.warning("No digit regions detected.")
        return

    st.success(
        f"Detected regions: {len(regions)}"
    )

    # -------------------------------------------------
    # GRID CROPPING PREVIEW
    # -------------------------------------------------

    if processing_mode == "5 x 2 Grid (10 samples)":

        st.subheader("🔍 Grid Cropping Preview")

        preview_columns = st.columns(5)

        for index, region_image in enumerate(regions):

            with preview_columns[index % 5]:

                st.image(
                    region_image,
                    caption=f"Sample {index + 1}",
                    width=120
                )

    # -------------------------------------------------
    # PREDICT EACH DIGIT
    # -------------------------------------------------

    results = []
    predicted_sequence = ""

    for index, region_image in enumerate(regions):

        try:

            digit, confidence, probabilities = predict_digit(
                region_image
            )

            predicted_sequence += str(digit)

            save_prediction(
                image_name,
                f"Digit {index + 1}",
                digit,
                confidence,
                image_hash
            )

            results.append({

                "index": index + 1,
                "image": region_image,
                "digit": digit,
                "confidence": confidence,
                "probabilities": probabilities

            })

        except ValueError as error:

            st.warning(
                f"Digit {index + 1}: {error}"
            )

        except Exception as error:

            st.error(
                f"Could not process digit {index + 1}: {error}"
            )

    if not results:

        st.warning("No valid digit predictions were produced.")
        return

    # -------------------------------------------------
    # DISPLAY INDIVIDUAL PREDICTIONS
    # -------------------------------------------------

    st.subheader("Individual Digit Predictions")

    # Use 5 columns for grid mode
    number_of_columns = (
        5 if processing_mode == "5 x 2 Grid (10 samples)"
        else 3
    )

    columns = st.columns(number_of_columns)

    for index, result in enumerate(results):

        with columns[index % number_of_columns]:

            st.image(
                result["image"],
                width=110
            )

            st.markdown(
                f"""
                **Digit {result['index']}**

                <div class="result-box">
                    <div class="digit-number">
                        {result['digit']}
                    </div>
                    Confidence: {result['confidence']:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

    # -------------------------------------------------
    # RECOGNIZED NUMBER SEQUENCE
    # -------------------------------------------------

    st.divider()

    st.subheader("Recognized Number Sequence")

    st.markdown(
        f"""
        <div class="result-box">
            <div class="digit-number">
                {predicted_sequence}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if processing_mode == "5 x 2 Grid (10 samples)":

        if len(results) == 10:
            st.success(
                "All 10 grid samples were processed."
            )
        else:
            st.warning(
                f"Only {len(results)} of the 10 grid samples "
                "were successfully predicted."
            )


# =====================================================
# 13. MAIN TITLE
# =====================================================

st.markdown(
    '<div class="main-title">🔢 Handwritten Digit Recognition</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">CNN-Based Handwritten Digit Recognition System</div>',
    unsafe_allow_html=True
)

st.write("")

st.info(
    "Upload handwritten digits, capture them using your camera, "
    "or automatically load images from the input_images folder."
)


# =====================================================
# 14. PROCESSING MODE
# =====================================================

st.header("⚙️ Image Processing Settings")

processing_mode = st.selectbox(
    "Choose how to detect digits:",
    [
        "Automatic Multiple Digit Detection",
        "5 x 2 Grid (10 samples)"
    ]
)

top_margin = 12
digit_start = 30

if processing_mode == "5 x 2 Grid (10 samples)":

    st.info(
        "Use this mode for Figure_1.png, which contains "
        "10 samples arranged in 5 columns and 2 rows."
    )

    top_margin = st.slider(
        "Remove title area from top (%)",
        min_value=0,
        max_value=35,
        value=12
    )

    digit_start = st.slider(
        "Skip label area inside each cell (%)",
        min_value=0,
        max_value=65,
        value=30
    )

st.divider()


# =====================================================
# 15. INPUT METHOD SELECTION
# =====================================================

input_method = st.radio(
    "Choose your image input method:",
    [
        "📁 Manual Upload",
        "📷 Camera Capture",
        "⚡ Automatic Image Loading"
    ],
    horizontal=True
)

st.divider()


# =====================================================
# 16. MANUAL IMAGE UPLOAD
# =====================================================

if input_method == "📁 Manual Upload":

    st.header("📁 Manual Image Upload")

    st.write(
        "Upload one digit, multiple separate images, "
        "or an image containing multiple digits."
    )

    uploaded_files = st.file_uploader(
        "Choose your images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.success(
            f"{len(uploaded_files)} image(s) uploaded!"
        )

        for uploaded_file in uploaded_files:

            st.divider()

            try:

                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                st.subheader(uploaded_file.name)

                display_multiple_results(
                    image,
                    uploaded_file.name,
                    processing_mode
                )

            except Exception as error:

                st.error(
                    f"Could not process {uploaded_file.name}: {error}"
                )


# =====================================================
# 17. CAMERA CAPTURE
# =====================================================

elif input_method == "📷 Camera Capture":

    st.header("📷 Camera Capture")

    st.write(
        "Capture handwritten digits using your camera."
    )

    camera_image = st.camera_input("Take a picture")

    if camera_image is not None:

        try:

            image = Image.open(
                camera_image
            ).convert("RGB")

            display_multiple_results(
                image,
                "Camera Capture",
                processing_mode
            )

        except Exception as error:

            st.error(
                f"Camera image processing failed: {error}"
            )


# =====================================================
# 18. AUTOMATIC IMAGE LOADING
# =====================================================

elif input_method == "⚡ Automatic Image Loading":

    st.header("⚡ Automatic Image Loading")

    st.write(
        "Images placed inside the input_images folder "
        "will be loaded automatically."
    )

    INPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    image_files = []

    for extension in ["*.png", "*.jpg", "*.jpeg"]:

        image_files.extend(
            INPUT_FOLDER.glob(extension)
        )

    image_files = sorted(image_files)

    if image_files:

        st.success(
            f"{len(image_files)} image(s) found!"
        )

        if st.button(
            "🚀 Process All Images",
            type="primary"
        ):

            for image_path in image_files:

                st.divider()

                st.subheader(image_path.name)

                try:

                    image = Image.open(
                        image_path
                    ).convert("RGB")

                    display_multiple_results(
                        image,
                        image_path.name,
                        processing_mode
                    )

                except Exception as error:

                    st.error(
                        f"Could not process {image_path.name}: {error}"
                    )

    else:

        st.warning(
            "No images found in the input_images folder."
        )

        st.info(
            "Place PNG, JPG or JPEG images inside "
            "the input_images folder."
        )


# =====================================================
# 19. PREDICTION HISTORY
# =====================================================

st.divider()

st.header("📊 Prediction History")

history = st.session_state.prediction_history

if history:

    history_df = pd.DataFrame(history)

    total_predictions = len(history_df)

    unique_images = history_df[
        "Image Name"
    ].nunique()

    average_confidence = history_df[
        "Confidence (%)"
    ].mean()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Predictions",
        total_predictions
    )

    col2.metric(
        "Unique Images",
        unique_images
    )

    col3.metric(
        "Average Confidence",
        f"{average_confidence:.2f}%"
    )

    st.subheader("Prediction Records")

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    if st.button("🗑️ Clear Prediction History"):

        st.session_state.prediction_history = []
        st.session_state.processed_images = set()

        st.rerun()

else:

    st.info(
        "No prediction history available yet. "
        "Process an image to see your prediction records here."
    )


# =====================================================
# 20. PREDICTION ANALYTICS
# =====================================================

if st.session_state.prediction_history:

    st.divider()

    st.header("📈 Prediction Analytics")

    analytics_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    digit_counts = (
        analytics_df["Predicted Digit"]
        .value_counts()
        .reindex(range(10), fill_value=0)
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Predictions by Digit")

        st.bar_chart(digit_counts)

    with col2:

        st.subheader("Confidence Distribution")

        st.bar_chart(
            analytics_df["Confidence (%)"]
        )

    most_common = digit_counts.idxmax()

    if digit_counts.max() > 0:

        st.success(
            f"Most frequently predicted digit: {most_common}"
        )




# =====================================================
# 21. MODEL PERFORMANCE DASHBOARD
# =====================================================

st.divider()

st.header("🧠 CNN Model Performance Dashboard")

st.write(
    "Visual evaluation of the trained CNN model "
    "using the MNIST dataset."
)

# -----------------------------------------------------
# MODEL PERFORMANCE METRICS
# -----------------------------------------------------

accuracy_col, info_col = st.columns(2)

with accuracy_col:

    st.metric(
        "MNIST Test Accuracy",
        "99.06%"
    )

with info_col:

    st.metric(
        "Number of Classes",
        "10 Digits (0–9)"
    )


# -----------------------------------------------------
# CONFUSION MATRIX
# -----------------------------------------------------

st.subheader("📊 Confusion Matrix")

confusion_path = (
    BASE_DIR / "development" / "confusion_matrix.png"
)

if confusion_path.exists():

    st.image(
        str(confusion_path),
        caption="CNN Confusion Matrix",
        use_container_width=True
    )

else:

    st.warning(
        "Confusion matrix image not found."
    )


# -----------------------------------------------------
# TRAINING ACCURACY AND LOSS
# -----------------------------------------------------

st.subheader("📈 Training Performance")

col1, col2 = st.columns(2)

accuracy_path = (
    BASE_DIR / "development" / "training_accuracy.png"
)

loss_path = (
    BASE_DIR / "development" / "training_loss.png"
)

with col1:

    st.write("Training Accuracy")

    if accuracy_path.exists():

        st.image(
            str(accuracy_path),
            use_container_width=True
        )

    else:

        st.warning(
            "Training accuracy graph not found."
        )


with col2:

    st.write("Training Loss")

    if loss_path.exists():

        st.image(
            str(loss_path),
            use_container_width=True
        )

    else:

        st.warning(
            "Training loss graph not found."
        )


# -----------------------------------------------------
# ACTUAL VS PREDICTED
# -----------------------------------------------------

st.subheader("🔍 Actual vs Predicted Results")

actual_path = (
    BASE_DIR / "development" / "actual_vs_predicted.png"
)

if actual_path.exists():

    st.image(
        str(actual_path),
        caption="Actual and Predicted Digit Comparison",
        use_container_width=True
    )

else:

    st.warning(
        "Actual vs Predicted image not found."
    )


# =====================================================
# 22. FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#9e9e9e;">

    **Handwritten Digit Recognition using CNN**

    Built with Python, TensorFlow and Streamlit

    </div>
    """,
    unsafe_allow_html=True
)



# =====================================================
# 21. FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#9e9e9e;">

    **Handwritten Digit Recognition using CNN**

    Built with Python, TensorFlow and Streamlit

    </div>
    """,
    unsafe_allow_html=True
)
