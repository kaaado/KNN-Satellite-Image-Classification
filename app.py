import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import os
import pandas as pd
import base64 

st.set_page_config(
    page_title="KNN Satellite Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Global Constants ---
IMAGE_SIZE = (64, 64)
HIST_BINS = 16
MODEL_PATH = "./model" 

# --- Load Models and Transformers ---
@st.cache_resource
def load_components():
    """Loads the KNN model, scaler, PCA transformer, and class names."""
    try:
        knn_model_file = os.path.join(MODEL_PATH, 'knn_eurosat_model.pkl')
        scaler_file = os.path.join(MODEL_PATH, 'scaler.pkl')
        pca_file = os.path.join(MODEL_PATH, 'pca.pkl')
        classes_file = os.path.join(MODEL_PATH, 'class_names.pkl')

        required_files = [knn_model_file, scaler_file, pca_file, classes_file]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
             raise FileNotFoundError(f"Missing files: {', '.join(missing_files)}")

        knn_model = joblib.load(knn_model_file)
        scaler = joblib.load(scaler_file)
        pca = joblib.load(pca_file)
        class_names = joblib.load(classes_file)

        if not hasattr(knn_model, 'predict_proba') or \
           not hasattr(scaler, 'transform') or \
           not hasattr(pca, 'transform') or \
           not isinstance(class_names, list):
            raise TypeError("Loaded components seem invalid.")

        return knn_model, scaler, pca, class_names

    except FileNotFoundError as e:
        st.error(f"❌ Model file error: {e}. Please ensure the following files are in the directory '{MODEL_PATH}':")
        st.error("'knn_eurosat_model.pkl', 'scaler.pkl', 'pca.pkl', 'class_names.pkl'")
        st.error("💡 Run the Jupyter Notebook first to generate these files.")
        st.stop()
    except Exception as e:
        st.error(f"❌ An error occurred loading model components: {e}")
        st.stop()

# --- Load components ---
knn_model, scaler, pca, class_names = load_components()


# --- Simple CSS Styling (Keep as is) ---
st.markdown("""
<style>
/* Main container slightly off-white */
.main .block-container {
    background-color: #FFFFFF;
    padding: 2rem;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}
/* Sidebar styling */
[data-testid="stSidebar"] > div:first-child {
    background-color: #F8F9FA;
    padding: 1.5rem 1rem;
    border-right: 1px solid #E0E0E0;
}
/* Headers */
h1, h2, h3 { color: #333333; }
/* Text & Labels */
p, .stMarkdown, label, .stRadio > label, .stSelectbox > label { color: #555555; }
/* --- Sidebar Navigation Buttons --- */
[data-testid="stSidebar"] .stButton > button {
    display: block; width: 100%; text-align: left; margin-bottom: 0.5rem;
    padding: 0.6rem 0.8rem; font-weight: 500; border-radius: 6px; border: none;
    background-color: transparent; color: #333333;
    transition: background-color 0.2s ease-in-out, color 0.2s ease-in-out;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #E9ECEF; color: #007bff;
}
/* Regular Buttons */
.stButton:not([data-testid="stSidebar"] .stButton) > button {
    border-radius: 8px; border: 1px solid #007bff; background-color: #007bff;
    color: white; padding: 0.5rem 1rem;
}
.stButton:not([data-testid="stSidebar"] .stButton) > button:hover {
    background-color: #0056b3; border-color: #0056b3;
}
/* File uploader */
.stFileUploader > label { border: 1px dashed #ced4da; border-radius: 8px; padding: 1rem; }
.stFileUploader > label:hover { border-color: #007bff; }
/* Alert Boxes */
.stAlert { border-radius: 8px; border-left-width: 5px; margin-top: 1rem; margin-bottom: 1rem; }
/* Webcam Video & Uploaded Image */
.stVideo video, div[data-testid="stImage"] img { border-radius: 10px; border: 1px solid #E0E0E0; }
</style>
""", unsafe_allow_html=True)


# --- Helper Functions ---
def is_image_file(filename):
    ext = filename.lower().split('.')[-1]
    return ext in ('jpg','jpeg','png','tif','tiff','bmp','webp') # Added webp

def extract_features_from_image(img, hist_bins=16):
    # (Keep the function as is)
    chans = cv2.split(img)
    hist_features = []
    for ch in chans:
        ch_uint8 = cv2.normalize(ch, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        hist = cv2.calcHist([ch_uint8], [0], None, [hist_bins], [0,256])
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-8)
        hist_features.append(hist)
    hist_features = np.concatenate(hist_features)
    means = [ch.mean() for ch in chans]
    stds  = [ch.std()  for ch in chans]
    feat = np.concatenate([hist_features, means, stds])
    return feat

def validate_satellite_image(img_array_rgb):
    # (Keep the function as is)
    if img_array_rgb is None: return False, "❌ Image array is None."
    if not (img_array_rgb.min() >= 0 and img_array_rgb.max() <= 255): return False, f"⚠️ Pixel values out of range [0, 255]: min={img_array_rgb.min()}, max={img_array_rgb.max()}"
    try:
        mean_val = np.mean(img_array_rgb)
        std_val  = np.std(img_array_rgb)
    except Exception as e: return False, f"❌ Error processing image stats: {e}"
    if img_array_rgb.ndim != 3 or img_array_rgb.shape[2] != 3: return False, f"❌ Image not 3-channel RGB (ndim={img_array_rgb.ndim}, channels={img_array_rgb.shape[-1]})."
    if not (5 < mean_val < 250): return False, f"⚠️ Unusual average brightness: mean={mean_val:.1f}"
    if std_val < 5: return False, f"⚠️ Very low variance: std={std_val:.1f}"
    return True, f"✅ Validated (mean={mean_val:.1f}, std={std_val:.1f})"

# --- MODIFIED PREPROCESSING FUNCTION ---
@st.cache_data # Cache still useful for performance if *same file* is re-uploaded
def preprocess_image_bytes_for_knn(_image_bytes, _scaler, _pca):
    """Decodes image bytes and preprocesses for the KNN+PCA model."""
    try:
        # 1. Decode image bytes
        nparr = np.frombuffer(_image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
             raise ValueError("Could not decode image bytes.")
        img_array_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2. Validate (using the decoded array)
        is_valid, msg = validate_satellite_image(img_array_rgb)
        if not is_valid:
            return None, f"Image validation failed: {msg}", None # Return original array for display

        # 3. Resize
        img_resized = cv2.resize(img_array_rgb, IMAGE_SIZE)

        # 4. Median Blur
        img_filtered = cv2.medianBlur(img_resized, 3)

        # 5. Extract Features
        features = extract_features_from_image(img_filtered, hist_bins=HIST_BINS)
        features = features.reshape(1, -1).astype(np.float32)

        # 6. Scale Features
        features_scaled = _scaler.transform(features)

        # 7. Apply PCA
        features_pca = _pca.transform(features_scaled)

        # Return PCA features, success message, and the *original decoded* array for display consistency
        return features_pca, "Preprocessing successful", img_array_rgb

    except ValueError as e:
        error_msg = f"❌ Error during preprocessing (PCA mismatch or decode error?): {e}"
        print(error_msg)
        # Try to return original array if decoding worked, else None
        img_to_display = img_array_rgb if 'img_array_rgb' in locals() else None
        return None, error_msg, img_to_display
    except Exception as e:
        error_msg = f"❌ Unexpected error during preprocessing: {e}"
        print(error_msg)
        img_to_display = img_array_rgb if 'img_array_rgb' in locals() else None
        return None, error_msg, img_to_display


# --- Realtime Video Processing Class (Requires similar modification) ---
class VideoProcessor:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.last_pred = "Initializing..."
        self.last_conf = 0.0

    # @st.cache_data - Caching inside class methods used by webrtc is tricky, avoid for now
    def process_frame_for_knn(self, img_rgb):
         # This function now does the full pipeline *without* Streamlit caching
         # It mirrors preprocess_image_bytes_for_knn but starts with an array
        try:
            is_valid, msg = validate_satellite_image(img_rgb)
            if not is_valid: return None, f"Validation failed: {msg}"
            img_resized = cv2.resize(img_rgb, IMAGE_SIZE)
            img_filtered = cv2.medianBlur(img_resized, 3)
            features = extract_features_from_image(img_filtered, hist_bins=HIST_BINS)
            features = features.reshape(1, -1).astype(np.float32)
            features_scaled = scaler.transform(features) # Use global scaler
            features_pca = pca.transform(features_scaled) # Use global pca
            return features_pca, "Success"
        except Exception as e:
            print(f"Error in process_frame_for_knn: {e}")
            return None, f"Error: {e}"


    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- Process Frame for KNN ---
        processed_features, msg = self.process_frame_for_knn(img_rgb)
        pred_class = "Processing..."
        confidence = 0.0

        if processed_features is not None:
            try:
                proba = knn_model.predict_proba(processed_features)
                pred_index = np.argmax(proba)
                confidence = proba[0, pred_index] * 100
                pred_class = class_names[pred_index]
            except Exception as e:
                print(f"Error during KNN prediction: {e}")
                pred_class = "Predict Error"
                confidence = 0.0
        else:
            # Shorten message for display if it's long
            pred_class = "Invalid Frame" if "Validation failed" in msg else "Preproc Error"
            confidence = 0.0

        self.last_pred = pred_class
        self.last_conf = confidence
        # --- End Processing ---

        text = f"{self.last_pred} ({self.last_conf:.1f}%)"
        if self.last_conf < 40: color = (0, 0, 255) # Red
        elif self.last_conf < 70: color = (0, 255, 255) # Yellow
        else: color = (0, 255, 0) # Green

        (text_width, text_height), baseline = cv2.getTextSize(text, self.font, 1.0, 2)
        rect_start = (5, 5)
        rect_end = (rect_start[0] + text_width + 10, rect_start[1] + text_height + 10)
        try:
            sub_img = img[rect_start[1]:rect_end[1], rect_start[0]:rect_end[0]]
            black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
            res = cv2.addWeighted(sub_img, 0.5, black_rect, 0.5, 1.0)
            img[rect_start[1]:rect_end[1], rect_start[0]:rect_end[0]] = res
        except Exception as e:
            print(f"Error drawing text background: {e}")
        text_pos = (rect_start[0] + 5, rect_start[1] + text_height + 5)
        cv2.putText(img, text, text_pos, self.font, 1.0, color, 2, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- Sidebar Navigation & State (Keep as is) ---
st.sidebar.title("🛰️ Navigation")
if 'app_mode' not in st.session_state: st.session_state.app_mode = "Image Classifier"
if st.sidebar.button("🏠 Image Classifier", key="nav_classifier", use_container_width=True): st.session_state.app_mode = "Image Classifier"
if st.sidebar.button("📊 Dataset Info", key="nav_dataset", use_container_width=True): st.session_state.app_mode = "Dataset Info"
if st.sidebar.button("ℹ️ About", key="nav_about", use_container_width=True): st.session_state.app_mode = "About"
st.sidebar.markdown("---")
app_mode = st.session_state.app_mode


# --- Main Content Area ---

if app_mode == "Image Classifier":
    st.title("🛰️ KNN Satellite Image Classifier")
    st.markdown("Upload an image or use the live webcam feed for classification.")
    st.markdown("---")

    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("**Model:** KNN (Optimized w/ PCA)")
    source_choice = st.sidebar.radio(
        "Select Input Source:",
        ["Upload Image", "Live Webcam"],
        horizontal=True,
        key="source_select"
    )

    if source_choice == "Upload Image":
        st.header("🖼️ Upload Image")
        uploaded_file = st.file_uploader(
            "Drag & drop a satellite image here OR click to browse...",
            type=["jpg", "jpeg", "png", "webp"], # Added webp
            label_visibility="collapsed",
            key="file_uploader" # Added a key
        )

        col1, col2 = st.columns([0.6, 0.4])

        if uploaded_file is not None:
            # Get image bytes ONCE
            image_bytes = uploaded_file.getvalue()
            try:
                # --- Call MODIFIED preprocessing function ---
                # Pass bytes, scaler, pca
                processed_features, prep_msg, display_image_array = preprocess_image_bytes_for_knn(
                    image_bytes, scaler, pca
                )

                with col1:
                    if display_image_array is not None:
                        st.image(display_image_array, caption=f'Uploaded: {uploaded_file.name}', use_column_width='auto')
                    else:
                        st.warning("Could not display image due to decoding error.")

                with col2:
                    st.subheader("📊 Prediction Results")
                    st.info(f"Image Status: {prep_msg}") # Display validation/preprocessing message

                    # Only predict if preprocessing was successful
                    if processed_features is not None:
                        with st.spinner("Classifying..."):
                            proba = knn_model.predict_proba(processed_features)[0]
                            top_indices = np.argsort(proba)[::-1][:3]
                            main_pred_index = top_indices[0]
                            pred_class = class_names[main_pred_index]
                            confidence = proba[main_pred_index] * 100

                            st.success(f"**Predicted Class:** `{pred_class}`")
                            st.metric(label="Confidence", value=f"{confidence:.2f}%")

                            st.markdown("**Top 3 Probabilities:**")
                            for i in top_indices:
                                st.write(f"- `{class_names[i]}`: {proba[i]*100:.2f}%")

                            proba_df = pd.DataFrame({
                                'Class': class_names,
                                'Probability': proba * 100
                            }).sort_values('Probability', ascending=False)

                            with st.expander("📊 Show Full Probability Distribution"):
                                st.bar_chart(proba_df.set_index('Class'))
                    # If preprocessing failed, message is already shown via prep_msg
                    elif "Validation failed" in prep_msg:
                         st.error("Classification aborted due to image validation failure.")
                    else: # Other preprocessing error
                         st.error("Classification aborted due to preprocessing error.")

            except Exception as e:
                st.error(f"An error occurred displaying or processing the image: {e}")
                with col1: st.empty()
                with col2: st.info("Upload an image for results.")

        else:
            with col1:
                st.info("✨ Drag and drop an image or use the button above to start classifying! ✨")
            with col2:
                st.subheader("📊 Prediction Results")
                st.info("Results will appear here.")

    elif source_choice == "Live Webcam":
        st.header("📷 Live Webcam Feed")
        st.warning("⚠️ The model is trained on satellite images. Webcam results will likely be inaccurate!")
        RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        webrtc_streamer(
            key="webcam-streamer",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=VideoProcessor,
            async_processing=True,
        )

# --- Dataset Info and About Pages (Keep as is) ---
elif app_mode == "Dataset Info":
    st.title("📊 EuroSAT Dataset Information")
    st.markdown("---")
    st.header("Classes")
    st.write("The KNN model was trained to recognize the following 10 classes:")
    num_classes = len(class_names)
    cols = st.columns(3)
    for i, cls_name in enumerate(class_names):
        with cols[i % 3]:
            st.info(f"**{cls_name}**")
    st.markdown("---")
    st.header("About EuroSAT")
    st.markdown(
        """
        EuroSAT is based on Sentinel-2 satellite images (13 spectral bands),
        with 10 classes and 27,000 labeled images. This model uses the **RGB version**.
        *Source: Helber, P., et al. Eurosat: A novel dataset... JSTARS, 2019.*
        """
    )
    # Using a known stable URL for the example image
    st.image("https://github.com/phelber/eurosat/raw/master/eurosat_overview_small.jpg", caption="Example EuroSAT Images (Source: EuroSAT GitHub)", use_column_width=True)


elif app_mode == "About":
    st.title("ℹ️ About This Application")
    st.markdown("---")
    st.markdown(
        """
        Demonstrates satellite image classification using **KNN** trained on **EuroSAT (RGB)**.
        **Pipeline:** Load ➔ Resize/Blur ➔ Extract Features (Histogram, Mean/Std) ➔ Normalize ➔ PCA ➔ Train/Predict.
        Built with Streamlit, Scikit-learn, OpenCV for an Adv. ML project.
        """
    )
    st.markdown("---")
    st.caption("Developed by kaaado") 