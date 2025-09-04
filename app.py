import tempfile
import streamlit as st
import whisper
import asyncio
import subprocess
import nest_asyncio
from deep_translator import GoogleTranslator
from difflib import SequenceMatcher
import numpy as np
import time
from dotenv import load_dotenv
import os
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from pydub import AudioSegment
import soundfile as sf
import yt_dlp

nest_asyncio.apply()
load_dotenv()

# API keys
# os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Language options (ISO 639-1 codes)
LANG_OPTIONS = {
    "Auto-detect": None,
    "Spanish (es)": "es",
    "Portuguese (pt)": "pt",
    "Arabic (ar)": "ar",
    "Turkish (tr)": "tr",
    "Indonesian (id)": "id"
}


output_path = "output.wav"
output_transcript_file_path = "transcript.txt"

# Whisper model
@st.cache_resource
def load_whisper_model(model_choice: str):
    if model_choice == "turbo":
        return whisper.load_model("turbo")
    elif model_choice == "large":
        return whisper.load_model("large-v3")
    else:
        return whisper.load_model("medium")  
    
 # UI for model choice
model_choice = st.selectbox(
    "Choose Whisper model:",
    ["medium", "large", "turbo"],
    index=None,  # ensures nothing is pre-selected
    placeholder="Select a model..."
)  
model = None
if model_choice:    
    model = load_whisper_model(model_choice=model_choice)

# --- Helpers ---
def format_timestamp(seconds):
    millis = int((seconds % 1) * 1000)
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02}:{seconds:02}.{millis:03}"

def compute_confidence(segment):
    lp = segment.get("avg_logprob", -5.0)
    lp_score = max(min((lp - (-6.0)) / (0.0 - (-6.0)), 1.0), 0.0)
    speech_factor = 1.0 - segment.get("no_speech_prob", 0.0)
    return float(max(0.0, min(1.0, 0.8 * lp_score + 0.2 * speech_factor)))

def compute_overall_confidence(result):
    segments = result.get("segments", [])
    if not segments:
        return 0.0
    total_duration = sum(max(0.001, seg["end"] - seg["start"]) for seg in segments)
    weighted = sum(compute_confidence(seg) * max(0.001, seg["end"] - seg["start"]) for seg in segments)
    return float(weighted / total_duration)

def label_confidence(score: float) -> str:
    if score >= 0.70: return "High ✅"
    elif score >= 0.40: return "Medium ⚠️"
    else: return "Low ❌"

async def download_m3u8_audio_async(m3u8_url):
    process = await asyncio.create_subprocess_exec(
        "ffmpeg", 
        "-y",
        "-i", m3u8_url,
        "-vn", 
        "-ar", "16000", "-ac", "1",
        "-c:a", "pcm_s16le",
        "-af", "highpass=f=200, lowpass=f=3000, dynaudnorm",
        output_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    return output_path if process.returncode == 0 else None

def find_all_word_timestamps(result, target_word):
    timestamps = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            if target_word.lower() in w["word"].lower():
                timestamps.append(w["start"])
    return timestamps

def generate_video_clip(start_time, m3u8_url, clip_file, duration=15, offset=5):
    # Ensure we don't go before 0
    adjusted_start = max(0, start_time - offset)

    cmd = [
        "ffmpeg", "-y", "-ss", str(adjusted_start),
        "-i", m3u8_url, "-t", str(duration),
        "-c", "copy", clip_file
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return clip_file

def split_audio(file_path, chunk_length=60):
    """
    Splits audio into chunks of given length (seconds).
    Returns a list of temporary file paths.
    """
    audio = AudioSegment.from_file(file_path)
    duration_ms = len(audio)
    chunks = []

    for i in range(0, duration_ms, chunk_length * 1000):
        chunk = audio[i:i + chunk_length * 1000]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        chunk.export(temp_file.name, format="wav")
        chunks.append(temp_file.name)

    return chunks

def merge_results(results):
    """
    Merge Whisper transcription results (chunked).
    """
    merged = {"text": "", "segments": [], "language": None}
    for r in results:
        merged["text"] += r["text"] + " "
        merged["segments"].extend(r["segments"])
        if merged["language"] is None:
            merged["language"] = r.get("language")
    return merged

def validate_audio(file_path: str):
    """Check if audio exists and is valid before Whisper transcribe"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if os.path.getsize(file_path) == 0:
        raise ValueError("Audio file is empty (0 bytes).")

    try:
        f = sf.SoundFile(file_path)
        if len(f) == 0:
            raise ValueError("Audio file has no frames (empty).")
    except Exception as e:
        raise ValueError(f"Invalid audio file: {e}")

# Read the transcript file
def read_transcript(file_path, results):
    # Write transcript to file
    with open(file_path, "w", encoding="utf-8") as file:
        for segment in results["segments"]:
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            text = segment["text"].strip()
            confidence = compute_confidence(segment)

            subtitle_line = (
                f"[{start_time} --> {end_time}]  {text}  "
                f"(Conf: {confidence:.2f})\n"
            )
            file.write(subtitle_line)

    # Now read and return content
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()
    
def split_text(text,max_length=4000):
    """Splits text into smaller chunks of max_length characters."""
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]    

def translate_to_english(file_path, source_lang="auto"):
    """
    Reads text from a file and translates it into English using GoogleTranslator.

    :param file_path: Path to the transcript file.
    :param source_lang: The source language (default is 'auto' for auto-detection).
    :return: Translated text in English.
    """
    # Read text from the file
    text = read_transcript(file_path,st.session_state.result) 
    text_chunks = split_text(text)

    def compute_translation_confidence(original_text: str, translated_text: str, source_lang="auto") -> float:
        """
        Estimates translation confidence by round-trip translation similarity.
        """
        try:
            # Back translate English -> source
            back_translator = GoogleTranslator(source="en", target=source_lang)
            back_translated = back_translator.translate(translated_text)

            # Compare original vs back-translated
            similarity = SequenceMatcher(None, original_text, back_translated).ratio()

            # Clamp between 0–1
            return float(max(0.0, min(1.0, similarity)))
        except Exception as e:
            print(f"Translation confidence failed: {e}")
            return 0.0

    with st.status("Translating in English", expanded = True) as status:
        try:
            translator = GoogleTranslator(source=source_lang, target="en")
            translated_chunks = [translator.translate(chunk) for chunk in text_chunks] 
            st.session_state.english_translated_text  = " ".join(translated_chunks)
            st.text_area("English Translation", st.session_state.english_translated_text, height = 500)
            # Compute translation confidence (per chunk or overall)
            confidence_scores = [
                compute_translation_confidence(orig, trans, source_lang=source_lang)
                for orig, trans in zip(text_chunks, translated_chunks)
            ]
           # overall_conf = np.mean(confidence_scores) if confidence_scores else 0.0

            #st.write(f"🔹 **Translation Confidence:** {overall_confidence:.2f} → {label_confidence(overall_confidence)}")

            status.update(label="Translated in English",state="complete")
        except Exception as e:
            st.error(f"Translation failed: {str(e)}") 
            translated_text = None   

def safe_transcribe(model, audio_path, **kwargs):
    """
    Wrapper around Whisper's transcribe with safety checks:
    - Ensures file exists and is not empty
    - Ensures duration > 0
    - Catches zero-length mel issues
    """

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if os.path.getsize(audio_path) == 0:
        raise ValueError(f"Audio file {audio_path} is empty.")

    try:
        info = sf.info(audio_path)
        duration = info.frames / info.samplerate
        if duration <= 0:
            raise ValueError(f"Audio file {audio_path} has zero duration.")
    except RuntimeError:
        raise ValueError(f"Cannot read audio metadata: {audio_path}")

    # Run Whisper transcription
    try:
        result = model.transcribe(audio_path, **kwargs)
    except RuntimeError as e:
        if "cannot reshape tensor of 0 elements" in str(e):
            raise ValueError(f"Whisper received empty audio features from {audio_path}. Check input.")
        else:
            raise

    # Handle completely silent audio gracefully
    if not result.get("text") or result["text"].strip() == "":
        result["text"] = "[No speech detected]"

    return result   

def download_youtube_wav(youtube_url: str, filename: str = "output") -> str:
    """
    Download YouTube audio directly as WAV and return the saved path.
    """


    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": filename,  # force final name
        "quiet": True,
        "noplaylist": True,  # avoid downloading full playlist accidentally
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",   # directly to wav
                "preferredquality": "192",
            }
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    return output_path
         

if model:            
    # --- UI ---
    st.set_page_config(page_title="Transcript Analyzer", page_icon="🎙️", layout="wide")
    st.markdown("<h1 style='color:#4CAF50'>🎙️ Transcript Analyzer</h1>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📥 Upload", "📝 Transcript", "🌍 Translation", "💬 Q&A", "🎬 Clips"])

    # --- Upload Tab ---
    with tab1:
        st.markdown("### Enter Video URL")
        url = st.text_input("M3U8 or Youtube Video URL", "")
        selected_lang = st.selectbox("Select Language", list(LANG_OPTIONS.keys()))

        if st.button("Download Audio"):
            if url:
                with st.status("Downloading Audio...", expanded=True) as status:
                    st.session_state.m3u8_url = url
                    loop = asyncio.get_event_loop()
                    if "youtube.com" in url or "youtu.be" in url:
                        result = download_youtube_wav(url)
                        st.success("youtube download started")
                    else:
                        result = loop.run_until_complete(download_m3u8_audio_async(url))
                    if result:
                        status.update(label="Audio Downloaded", state="complete")
                        st.success("Audio downloaded successfully!")
                    else:
                        status.update(label="Failed to download audio", state="error")
                        st.error("Failed to download audio. Please check the URL.")
                        
            else:
                st.error("Please enter a valid URL")

        if st.button("Generate Transcript"):
            if url:
                # with st.status("Downloading Audio...", expanded=True) as status:
                #     st.session_state.m3u8_url = m3u8_url
                #     loop = asyncio.get_event_loop()
                #     result = loop.run_until_complete(download_m3u8_audio_async(m3u8_url))
                #     status.update(label="Audio Downloaded", state="complete")

                with st.status("Generating Transcript...", expanded=True) as status:
                    #validate_audio(output_path)  
                    result = safe_transcribe(model= model, audio_path=output_path, language= LANG_OPTIONS[selected_lang],verbose=True)
                    #result = model.transcribe(output_path, language= LANG_OPTIONS[selected_lang],verbose=True)
                    st.session_state.result = result
                    status.update(label="Transcript Generated",state="complete")

                st.success(" Transcripts generated!")
            else:
                st.error("Please enter a valid URL")

    # --- Transcript Tab ---
    with tab2:
        st.markdown("### 📜 Transcript Viewer")
        if "result" in st.session_state:
            col1, = st.columns(1)
            with col1:
                with st.expander("Transcript"):
                    st.text_area("Transcript", read_transcript(output_transcript_file_path,st.session_state.result), height= 500)
                st.info(f"Confidence: {compute_overall_confidence(st.session_state.result):.2f} → {label_confidence(compute_overall_confidence(st.session_state.result))}")
        else:
            st.warning("No transcript available. Please generate one first.")



    # --- Translation Tab ---
    with tab3:
        st.markdown("### 🌍 Google Translator (Optional)")
        
        if "result" in st.session_state:
            try:
                text = " ".join([seg["text"] for seg in st.session_state.result["segments"]])
                with st.spinner("Translating with Google..."):
                    translate_to_english(output_transcript_file_path, source_lang="auto")
            except Exception as e:
                st.error(f"Translation failed: {str(e)}")
        else:
            st.warning("Generate transcript first.")

    # --- Q&A Tab ---
    with tab4:
        st.markdown("### 💬 Ask Questions")
        if "english_translated_text" in st.session_state:
            llm = ChatGroq( groq_api_key=groq_api_key,
                            model_name="llama-3.3-70b-versatile",
                            # optional params you may want to configure:
                            temperature=0.0,
                            max_tokens=1024,)
            prompt = ChatPromptTemplate.from_template(
                """
                Answer the questions based on the provided context only.
                Please provide the most accurate respone based on the question
                <context>
                {context}
                <context>
                Question:{input}

                """
            )

            if "vectors" not in st.session_state :
                docs = [Document(page_content=st.session_state.english_translated_text)]
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
                final_docs = text_splitter.split_documents(docs)
                st.session_state.embeddings = OpenAIEmbeddings()
                st.session_state.vectors = FAISS.from_documents(final_docs, st.session_state.embeddings)

            query = st.chat_input("Ask something about the transcript...")
                # Show user’s question in Streamlit chat.
                # Build a retriever from a vector store.
                # Create a doc-processing chain with an LLM and prompt.
                # Combine retriever + LLM into a retrieval chain (RAG).
                # Run the chain with the user query → get the AI’s response.
            if query:
                st.chat_message("user").write(query)
                retriever = st.session_state.vectors.as_retriever()
                document_chain = create_stuff_documents_chain(llm, prompt)
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                response = retrieval_chain.invoke({'input': query})
                st.chat_message("assistant").write(response['answer'])
        else:
            st.warning("Generate English transcript first.")

    # --- Clips Tab ---
    with tab5:
        st.markdown("### 🎬 Generate Clips")

        # Run translation only when user lands in tab5
        if "result_en" not in st.session_state:
            if os.path.exists(output_path) and "result" in st.session_state:
                if st.button("Generate English Transcript for Clips"):
                    try:
                        with st.spinner("Building context to generate clips..."):

                            result_en = model.transcribe(
                                output_path,
                                task="translate",
                                word_timestamps=True,
                                verbose=True
                            )
                        if result_en is None:
                            st.error("Whisper returned None — check if audio exists and is valid.")
                        else:
                            st.session_state.result_en = result_en
                            st.success("English transcript generated for clips!")
                    except Exception as e:
                        st.error(f"Failed: {e}")    
            else:
                st.warning("Generate transcript first before creating clips.")

        if "result_en" in st.session_state:
            word = st.text_input("Enter a word (English transcript) to clip:")
            duration = st.slider("Clip Duration (seconds)", 5, 30, 15)
            if st.button("Generate Clips"):
                timestamps = find_all_word_timestamps(st.session_state.result_en, word)
                if timestamps:
                    st.success(f"Found {len(timestamps)} occurrence(s) of '{word}'")
                    for i, ts in enumerate(timestamps, start=1):
                        clip_file = f"clip_{word}_{i}.mp4"
                        generate_video_clip(ts, st.session_state.m3u8_url, clip_file, duration)
                        with st.container():
                            st.markdown(f"🎬 **Clip {i}** ({ts:.2f}s → {ts+duration:.2f}s)")
                            st.video(clip_file)
                else:
                    st.error(f"No occurrences of '{word}' found.")
        else:
            st.warning("Generate English transcript first.")
else :
    st.info("Please select a Whisper model to begin.")            
