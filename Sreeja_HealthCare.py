import streamlit as st
import pandas as pd
from pathlib import Path
import io

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Healthcare AI Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.html("""
<style>

body {
    background-color: #f4f8fb;
}

.main {
    background-color: #f4f8fb;
}

.block-container {
    max-width: 1250px;
    padding-top: 30px;
    padding-bottom: 50px;
}


/* =========================
   SIDEBAR
========================= */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #e8f5ff,
        #f7fbfd
    );
}

.sidebar-logo {
    font-size: 25px;
    font-weight: 800;
    color: #087f5b;
    margin-bottom: 35px;
}

.sidebar-title {
    font-size: 14px;
    font-weight: 700;
    color: #334e68;
    margin-top: 25px;
    margin-bottom: 10px;
}

.sidebar-description {
    color: #52667a;
    font-size: 14px;
    line-height: 1.7;
}

.feature {
    color: #52667a;
    font-size: 14px;
    margin: 10px 0;
}


/* =========================
   HERO
========================= */

.hero {
    background: linear-gradient(
        135deg,
        #087f5b,
        #0b7285
    );

    padding: 45px;

    border-radius: 24px;

    color: white;

    margin-bottom: 30px;

    box-shadow:
        0 12px 30px rgba(8, 127, 91, 0.18);
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 10px;
}

.hero-subtitle {
    font-size: 18px;
    opacity: 0.92;
    margin-bottom: 22px;
}

.hero-badge {
    display: inline-block;

    padding: 9px 17px;

    border-radius: 30px;

    background: rgba(255,255,255,0.18);

    border: 1px solid rgba(255,255,255,0.25);

    font-size: 14px;

    font-weight: 600;
}


/* =========================
   FEATURE CARDS
========================= */

.info-card {
    background: white;

    padding: 25px;

    border-radius: 18px;

    min-height: 165px;

    border: 1px solid #e1ebf0;

    box-shadow:
        0 6px 20px rgba(0,0,0,0.06);
}

.info-icon {
    font-size: 32px;
    margin-bottom: 10px;
}

.info-title {
    font-size: 18px;
    font-weight: 750;
    color: #243b53;
    margin-bottom: 8px;
}

.info-text {
    font-size: 14px;
    color: #627d98;
    line-height: 1.6;
}


/* =========================
   CHAT HEADER
========================= */

.chat-header {
    background: white;

    padding: 25px 30px;

    border-radius: 20px 20px 0 0;

    border: 1px solid #e1ebf0;

    margin-top: 30px;
}

.chat-title {
    font-size: 24px;

    font-weight: 800;

    color: #243b53;
}

.chat-subtitle {
    color: #829ab1;

    font-size: 14px;

    margin-top: 5px;
}


/* =========================
   USER MESSAGE
========================= */

.user-message {
    background: #e7f5ff;

    border-left: 4px solid #1971c2;

    padding: 18px 22px;

    border-radius: 5px 18px 18px 18px;

    margin: 15px 0;

    color: #243b53;
}

.user-label {
    color: #1864ab;

    font-weight: 800;

    font-size: 13px;

    margin-bottom: 8px;
}


/* =========================
   BOT MESSAGE
========================= */

.bot-message {
    background: #ebfbee;

    border-left: 4px solid #2f9e44;

    padding: 18px 22px;

    border-radius: 5px 18px 18px 18px;

    margin: 15px 0;

    color: #243b53;
}

.bot-label {
    color: #2b8a3e;

    font-weight: 800;

    font-size: 13px;

    margin-bottom: 8px;
}

.message-text {
    font-size: 15px;

    line-height: 1.7;
}


/* =========================
   CONFIDENCE
========================= */

.confidence-card {
    background: white;

    border: 1px solid #e1ebf0;

    padding: 18px 20px;

    border-radius: 15px;

    margin-top: 10px;
}

.confidence-label {
    color: #829ab1;

    font-size: 13px;

    font-weight: 600;
}

.confidence-value {
    color: #087f5b;

    font-size: 28px;

    font-weight: 800;

    margin-top: 3px;
}


/* =========================
   MATCHED QUESTION
========================= */

.match-card {
    background: #fff9db;

    border: 1px solid #ffe066;

    padding: 18px 20px;

    border-radius: 15px;

    margin-top: 10px;
}

.match-title {
    color: #e67700;

    font-size: 14px;

    font-weight: 800;

    margin-bottom: 8px;
}

.match-text {
    color: #5f3b00;

    font-size: 14px;

    line-height: 1.6;
}


/* =========================
   INPUT
========================= */

.stTextInput input {

    background-color: black !important;

    border: 1px solid #ccd9e0 !important;

    border-radius: 14px !important;

    min-height: 50px !important;

    padding: 12px 16px !important;

    font-size: 15px !important;
}

.stTextInput input:focus {

    border-color: #0b7285 !important;

    box-shadow:
        0 0 0 2px rgba(11,114,133,0.12) !important;
}


/* =========================
   BUTTON
========================= */

.stButton > button {

    width: 100%;

    min-height: 48px;

    border: none;

    border-radius: 13px;

    background: linear-gradient(
        135deg,
        #087f5b,
        #0b7285
    );

    color: white;

    font-size: 15px;

    font-weight: 700;

    transition: 0.25s;
}

.stButton > button:hover {

    transform: translateY(-2px);

    box-shadow:
        0 8px 20px rgba(8,127,91,0.25);
}


/* =========================
   DISCLAIMER
========================= */

.disclaimer {

    background: #fff5f5;

    border: 1px solid #ffc9c9;

    padding: 22px;

    border-radius: 17px;

    margin-top: 30px;

    color: #842029;

    font-size: 14px;

    line-height: 1.7;
}

.disclaimer-title {

    font-size: 16px;

    font-weight: 800;

    margin-bottom: 8px;
}


/* =========================
   FOOTER
========================= */

.footer {

    text-align: center;

    margin-top: 45px;

    padding: 25px;

    color: #829ab1;

    font-size: 13px;

    border-top: 1px solid #dce6eb;

    line-height: 1.7;
}



/* =========================
   VOICE & CAMERA
========================= */

.media-panel {
    background: white;
    border: 1px solid #e1ebf0;
    padding: 22px;
    border-radius: 18px;
    margin-top: 18px;
    margin-bottom: 18px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.05);
}

.media-title {
    color: #243b53;
    font-size: 18px;
    font-weight: 800;
    margin-bottom: 5px;
}

.media-subtitle {
    color: #627d98;
    font-size: 13px;
    line-height: 1.5;
    margin-bottom: 14px;
}

.media-status {
    background: #e7f5ff;
    border: 1px solid #a5d8ff;
    color: #1864ab;
    padding: 10px 14px;
    border-radius: 10px;
    font-size: 13px;
    margin-top: 10px;
}

.stAudioInput > div,
.stCameraInput > div {
    border-radius: 14px !important;
}

/* =========================
   MOBILE
========================= */

@media (max-width: 768px) {

    .hero {
        padding: 30px;
    }

    .hero-title {
        font-size: 30px;
    }

    .hero-subtitle {
        font-size: 15px;
    }

}

</style>
""")


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "datasets"

KNOWLEDGE_FILE = DATASET_PATH / "healthcare_knowledge.csv"

TRAINING_FILE = DATASET_PATH / "healthcare_training_v2.csv"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html("""
    <div class="sidebar-logo">
        🏥 Healthcare Chatbot
    </div>
    """)

    st.html("""
    <div class="sidebar-title">
        🤖 AI Healthcare Assistant
    </div>

    <div class="sidebar-description">
        Your intelligent healthcare information
        assistant powered by Natural Language
        Processing and semantic search.
    </div>
    """)

    st.html("""
    <div class="sidebar-title">
        ✨ Features
    </div>

    <div class="feature">
        🔹 NLP-based question matching
    </div>

    <div class="feature">
        🔹 Semantic similarity search
    </div>

    <div class="feature">
        🔹 Healthcare knowledge base
    </div>

    <div class="feature">
        🔹 Confidence scoring
    </div>

    <div class="feature">
        🔹 Instant responses
    </div>

    <div class="feature">
        🔹 🎤 Voice questions
    </div>

    <div class="feature">
        🔹 📷 Live camera capture
    </div>

    <div class="feature">
        🔹 🖼️ Gallery image upload
    </div>
    """)

    st.divider()

    if st.button("🗑️ Clear Conversation"):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# HERO SECTION
# ============================================================

st.html("""
<div class="hero">

    <div class="hero-title">
        🏥 Healthcare AI Chatbot
    </div>

    <div class="hero-subtitle">
        Your intelligent healthcare information assistant
    </div>

    <span class="hero-badge">
        🤖 Powered by NLP & Semantic Search
    </span>

</div>
""")


# ============================================================
# FEATURE CARDS
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.html("""
    <div class="info-card">

        <div class="info-icon">
            🤖
        </div>

        <div class="info-title">
            AI Powered
        </div>

        <div class="info-text">
            Uses Sentence Transformers to
            understand the meaning of your
            healthcare questions.
        </div>

    </div>
    """)


with col2:

    st.html("""
    <div class="info-card">

        <div class="info-icon">
            ⚡
        </div>

        <div class="info-title">
            Instant Answers
        </div>

        <div class="info-text">
            Quickly searches the healthcare
            knowledge base and provides the
            most relevant response.
        </div>

    </div>
    """)


with col3:

    st.html("""
    <div class="info-card">

        <div class="info-icon">
            🔐
        </div>

        <div class="info-title">
            Knowledge Based
        </div>

        <div class="info-text">
            Responses are retrieved from
            your healthcare datasets using
            semantic similarity.
        </div>

    </div>
    """)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


with st.spinner("🤖 Loading AI model..."):

    model = load_model()


# ============================================================
# VOICE TRANSCRIPTION
# ============================================================

@st.cache_resource
def load_speech_recognizer():
    return sr.Recognizer()


def transcribe_audio(audio_file):
    """
    Convert Streamlit's WAV microphone recording into text.
    Uses Google's speech-recognition service through SpeechRecognition.
    """
    recognizer = load_speech_recognizer()

    try:
        audio_bytes = audio_file.getvalue()

        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data)
        return text.strip()

    except sr.UnknownValueError:
        return ""

    except sr.RequestError:
        return None

    except Exception as exc:
        st.error(f"❌ Could not process the audio: {exc}")
        return ""


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df1 = pd.read_csv(
        KNOWLEDGE_FILE
    )

    df2 = pd.read_csv(
        TRAINING_FILE
    )

    df = pd.concat(
        [df1, df2],
        ignore_index=True
    )

    required_columns = [
        "question",
        "response"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Column '{column}' is missing "
                f"from the dataset."
            )

    df = df.dropna(
        subset=[
            "question",
            "response"
        ]
    )

    df["question"] = (
        df["question"]
        .astype(str)
    )

    df["response"] = (
        df["response"]
        .astype(str)
    )

    df = df.drop_duplicates(
        subset=[
            "question",
            "response"
        ]
    )

    return df.reset_index(
        drop=True
    )


try:

    df = load_data()

except Exception as e:

    st.error(
        "❌ Error loading healthcare datasets."
    )

    st.exception(e)

    st.stop()


# ============================================================
# PREPARE DATA
# ============================================================

questions = df[
    "question"
].tolist()

responses = df[
    "response"
].tolist()


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

@st.cache_resource
def create_embeddings(
    questions
):

    return model.encode(
        questions,
        convert_to_tensor=False,
        show_progress_bar=False
    )


with st.spinner(
    "📚 Preparing healthcare knowledge..."
):

    question_embeddings = create_embeddings(
        questions
    )


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []

if "voice_question" not in st.session_state:

    st.session_state.voice_question = ""

if "camera_image" not in st.session_state:

    st.session_state.camera_image = None

if "uploaded_image_name" not in st.session_state:

    st.session_state.uploaded_image_name = ""


# ============================================================
# CHAT HEADER
# ============================================================

st.html("""
<div class="chat-header">

    <div class="chat-title">
        💬 Healthcare Assistant
    </div>

    <div class="chat-subtitle">
        Ask a healthcare-related question and
        get the most relevant information from
        the knowledge base.
    </div>

</div>
""")


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        st.html(f"""
        <div class="user-message">

            <div class="user-label">
                {("👤 You • " + message.get("input_type", "Text"))}
            </div>

            <div class="message-text">
                {message["content"]}
            </div>

        </div>
        """)

    else:

        st.html(f"""
        <div class="bot-message">

            <div class="bot-label">
                🤖 Healthcare AI Chatbot
            </div>

            <div class="message-text">
                {message["content"]}
            </div>

        </div>
        """)

        if "confidence" in message:

            confidence = message["confidence"]

            st.html(f"""
            <div class="confidence-card">

                <div class="confidence-label">
                    Semantic Similarity
                </div>

                <div class="confidence-value">
                    {confidence * 100:.2f}%
                </div>

            </div>
            """)

        if "matched_question" in message:

            st.html(f"""
            <div class="match-card">

                <div class="match-title">
                    🔎 Matched Knowledge
                </div>

                <div class="match-text">
                    {message["matched_question"]}
                </div>

            </div>
            """)


# ============================================================
# MULTIMODAL INPUT
# ============================================================

st.markdown("### Ask your question")

st.html("""
<div class="media-panel">
    <div class="media-title">🎤 📷 Multimodal Input</div>
    <div class="media-subtitle">
        Type your healthcare question, record it using your microphone,
        or use your camera to capture an image.
    </div>
</div>
""")

input_tab1, input_tab2, input_tab3 = st.tabs(
    ["⌨️ Text", "🎤 Voice", "📷 Camera"]
)

# -------------------------
# TEXT INPUT
# -------------------------

with input_tab1:

    user_question = st.text_input(
        "Healthcare question",
        placeholder="Example: What are the symptoms of diabetes?",
        label_visibility="collapsed",
        key="text_question"
    )

    st.html("""
    <div class="media-status">
        💡 Type your question and click <b>Ask Healthcare AI Chatbot</b>.
    </div>
    """)

# -------------------------
# VOICE INPUT
# -------------------------

with input_tab2:

    audio_value = st.audio_input(
        "🎤 Record your healthcare question",
        sample_rate=16000,
        key="healthcare_voice"
    )

    voice_question = ""

    if audio_value is not None:

        st.audio(audio_value)

        with st.spinner("🎙️ Converting your voice to text..."):

            voice_question = transcribe_audio(audio_value)

        if voice_question is None:

            st.error(
                "❌ Speech recognition service could not be reached. "
                "Please check your internet connection and try again."
            )

        elif voice_question:

            st.success(f"📝 Transcribed question: {voice_question}")

            st.session_state.voice_question = voice_question

        else:

            st.warning(
                "⚠️ I couldn't understand the recording. "
                "Please speak clearly and try again."
            )

    if st.session_state.get("voice_question"):

        st.info(
            f"🎤 Ready to ask: {st.session_state.voice_question}"
        )

# -------------------------
# CAMERA + GALLERY INPUT
# -------------------------

with input_tab3:

    image_source = st.radio(
        "Choose image source",
        ["📷 Take Live Picture", "🖼️ Upload from Gallery"],
        horizontal=True,
        key="image_source"
    )

    if image_source == "📷 Take Live Picture":

        st.markdown("#### 📷 Capture a live picture")

        camera_image = st.camera_input(
            "Take a picture",
            key="healthcare_camera",
            resolution="720p"
        )

        if camera_image is not None:

            st.image(
                camera_image,
                caption="📷 Captured image",
                use_container_width=True
            )

            st.session_state.camera_image = camera_image.getvalue()
            st.session_state.uploaded_image_name = "camera_capture.jpg"

            st.success(
                "✅ Picture captured successfully."
            )

        else:

            st.info(
                "📷 Allow camera access in your browser, then take a picture."
            )

    else:

        st.markdown("#### 🖼️ Upload a picture from your device")

        gallery_image = st.file_uploader(
            "Choose an image from your gallery",
            type=["jpg", "jpeg", "png", "webp"],
            key="healthcare_gallery",
            help="Upload a JPG, JPEG, PNG, or WEBP image."
        )

        if gallery_image is not None:

            st.image(
                gallery_image,
                caption=f"🖼️ {gallery_image.name}",
                use_container_width=True
            )

            st.session_state.camera_image = gallery_image.getvalue()
            st.session_state.uploaded_image_name = gallery_image.name

            st.success(
                f"✅ Image uploaded successfully: {gallery_image.name}"
            )

        else:

            st.info(
                "🖼️ Select an image from your phone, computer, or gallery."
            )

    # Common status message.
    if st.session_state.get("camera_image"):

        st.html("""
        <div class="media-status">
            ✅ An image is ready. You can replace it anytime by taking
            another picture or selecting a different gallery image.
        </div>
        """)


# ============================================================
# ASK BUTTON
# ============================================================

if st.button(
    "🔍 Ask Healthcare AI Chatbot",
    key="ask_healthcare_button"
):

    # Select the active question.
    text_question = st.session_state.get("text_question", "").strip()
    voice_question = st.session_state.get("voice_question", "").strip()

    # Voice takes priority if a voice question has been recorded.
    if voice_question:
        active_question = voice_question
        input_type = "🎤 Voice"

    else:
        active_question = text_question
        input_type = "⌨️ Text"

    if not active_question:

        st.warning(
            "⚠️ Please type a question or record one using the microphone."
        )

    else:

        # Add user message.
        st.session_state.messages.append(
            {
                "role": "user",
                "content": active_question,
                "input_type": input_type
            }
        )

        # Search healthcare knowledge base.
        with st.spinner(
            "🤔 Understanding your question..."
        ):

            user_embedding = model.encode(
                [active_question],
                convert_to_tensor=False
            )

            scores = cosine_similarity(
                user_embedding,
                question_embeddings
            )

            index = scores[0].argmax()

            confidence = float(
                scores[0][index]
            )

            answer = responses[index]

            matched_question = questions[index]

        # Add bot message.
        st.session_state.messages.append(
            {
                "role": "bot",
                "content": answer,
                "confidence": confidence,
                "matched_question": matched_question
            }
        )

        # Clear the processed voice question so it isn't submitted again.
        st.session_state.voice_question = ""

        st.rerun()



# ============================================================
# MEDICAL DISCLAIMER
# ============================================================

st.html("""
<div class="disclaimer">

    <div class="disclaimer-title">
        ⚕️ Medical Disclaimer
    </div>

    This AI chatbot is designed for educational
    and informational purposes only. It does not
    provide professional medical diagnosis,
    treatment, or medical advice.

    <br><br>

    For serious symptoms or medical emergencies,
    please contact a qualified healthcare professional
    or emergency medical service immediately.

</div>
""")


# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="footer">

    🏥 <b>Healthcare AI Chatbot</b>

    <br>

    NLP-Based Healthcare Information Assistant

    <br><br>

    Built using Python • Streamlit • NLP • Sentence Transformers •
    Voice Input • Camera Input

</div>
""")