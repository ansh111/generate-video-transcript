from difflib import SequenceMatcher
import streamlit as st
import whisper
import asyncio
from deep_translator import GoogleTranslator
import nest_asyncio
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from langchain.chains import create_retrieval_chain
import time
import numpy as np
import subprocess

nest_asyncio.apply()


load_dotenv()
## load the GROQ API Key
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
groq_api_key= os.environ['GROQ_API_KEY']

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

output_path = "output.wav"
output_transcript_file_path = "transcript.txt"

# Load the Whisper model only once
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("large-v3")

# initilaise it only once 
model = load_whisper_model()
    
async def download_m3u8_audio_async(m3u8_url):
    """Downloads and converts an .m3u8 stream to .wav using ffmpeg asynchronously."""
    # process = await asyncio.create_subprocess_exec(
    #     "ffmpeg",
    #     "-y", 
    #     "-i", m3u8_url,
    #     "-q:a", "0",
    #     "-map", "a",
    #     output_path,
    #     stdout=asyncio.subprocess.PIPE,
    #     stderr=asyncio.subprocess.PIPE
    # )

    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-y", 
        "-i", m3u8_url,
        "-vn",                        # drop video
        "-ar", "16000",               # resample to 16kHz
        "-ac", "1",                   # mono
        "-c:a", "pcm_s16le",          # uncompressed wav (better than mp3/aac)
        "-af", "highpass=f=200, lowpass=f=3000, dynaudnorm",  # clean speech band + normalize
        output_path,                  # should be .wav
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    if process.returncode == 0:
        return output_path
    else:
        print(stderr.decode())  # Print error message
        return None    

# Streamlit UI
st.title("Transcribe Any Video")

# User Input
m3u8_url = st.text_input("Enter URL", "")
st.session_state.m3u8_url = m3u8_url

def format_timestamp(seconds):
    millis = int((seconds % 1) * 1000)
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02}:{seconds:02}.{millis:03}"

def _normalize_logprob(avg_logprob, min_lp=-6.0, max_lp=0.0):
    clamped = max(min(avg_logprob, max_lp), min_lp)
    return (clamped - min_lp) / (max_lp - min_lp)

def compute_confidence(segment):
    lp = segment.get("avg_logprob", -5.0)
    lp_score = _normalize_logprob(lp, min_lp=-6.0, max_lp=0.0)
    speech_factor = 1.0 - segment.get("no_speech_prob", 0.0)
    conf = 0.8 * lp_score + 0.2 * speech_factor
    return float(max(0.0, min(1.0, conf)))

def compute_overall_confidence(result):
    segments = result.get("segments", [])
    if not segments:
        return 0.0
    total_duration = sum(max(0.001, seg["end"] - seg["start"]) for seg in segments)
    weighted = sum(compute_confidence(seg) * max(0.001, seg["end"] - seg["start"]) for seg in segments)
    return float(weighted / total_duration)

def label_confidence(score: float) -> str:
    """Return a label for confidence value."""
    if score >= 0.70:
        return "High ✅"
    elif score >= 0.40:
        return "Medium ⚠️"
    else:
        return "Low ❌"


# Read the transcript file
def read_transcript(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()
    
if st.button("Generate Transcript"):
    if m3u8_url:
        with st.status("Downloading Audio...", expanded=True) as status:
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(download_m3u8_audio_async(m3u8_url))
            status.update(label="Audio Downloaded", state="complete")

        with st.status("Generating Transcript...", expanded=True) as status:
           result = model.transcribe(output_path, word_timestamps=True,verbose=True)
           st.session_state.result = result
           result_en = model.transcribe(
                output_path, 
                task="translate",       \
                word_timestamps=True, 
                verbose=True
            )
           st.session_state.result_en = result_en   # English transcript
           status.update(label="Transcript Generated",state="complete")

        with open(output_transcript_file_path, "w", encoding="utf-8") as file:
            for segment in result["segments"]:
                start_time = format_timestamp(segment["start"])
                end_time = format_timestamp(segment["end"])
                text = segment["text"].strip()
                confidence = compute_confidence(segment)

                subtitle_line = (
                    f"[{start_time} --> {end_time}]  {text}  "
                    f"(Conf: {confidence:.2f})\n"
                )
                file.write(subtitle_line)        
        
        with st.status("Showing Transcript", expanded=True) as status:
            st.text_area("Transcript", read_transcript(output_transcript_file_path), height= 500)
            status.update(label="Transcript Displayed",state="complete")

        with st.status("Overall Transcript Confidence", expanded=True) as status:
            overall_confidence = compute_overall_confidence(result)
            st.write(f"🔹 **Overall Transcript Confidence:** {overall_confidence:.2f} → {label_confidence(overall_confidence)}")
            st.markdown("""
            ### Legend:
            - ✅ **High (≥ 0.70)** — Transcript is reliable
            - ⚠️ **Medium (0.40 – 0.69)** — Some uncertainty, review suggested
            - ❌ **Low (< 0.40)** — Transcript likely has many errors
            """)
            status.update(label="Overall Transcript Confidence",state="complete")

    else:
        st.error("Please enter url")    

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
    text = read_transcript(file_path) 
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

           # st.write(f"🔹 **Translation Confidence:** {overall_conf:.2f} → {label_confidence(overall_conf)}")

            status.update(label="Translated in English",state="complete")
        except Exception as e:
            st.error(f"Translation failed: {str(e)}") 
            translated_text = None   



if st.button("Translate To English"):
    translate_to_english(output_transcript_file_path)


llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama3-70b-8192")

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate respone based on the question
    <context>
    {context}
    <context>
    Question:{input}

    """

)

def generate_transcript_embeddings():
     st.session_state.docs = [Document(page_content=st.session_state.english_translated_text)]
     st.write(st.session_state.docs) ## Document Loading
     st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
     st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
     st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)   

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.vectors = []
        st.session_state.embeddings=OpenAIEmbeddings()

    if not st.session_state.english_translated_text or not st.session_state.english_translated_text.strip():
        st.error("⚠️ english_translated_text is empty. Cannot create embeddings.")
        return    

    generate_transcript_embeddings()

if st.button("Add to Vector DB"):
    create_vector_embedding()

user_prompt=st.text_input("Enter your query from the transcript")

if st.button("Answer"):
    if user_prompt:
        document_chain = create_stuff_documents_chain(llm,prompt)
        retriever=st.session_state.vectors.as_retriever()
        retrieval_chain=create_retrieval_chain(retriever,document_chain)
        start=time.process_time()
        response=retrieval_chain.invoke({'input':user_prompt})
        print(f"Response time :{time.process_time()-start}")
        final_response =  response['answer']
        st.write(final_response)
        transcript = ""

    ## With a streamlit expander
        with st.expander("Document similarity Search"):
            for i,doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('------------------------')
    else:
        st.write("please enter a question to get an answer")   

# clip generstion code 
def find_all_word_timestamps(result, target_word):
    timestamps = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            if target_word.lower() in w["word"].lower():
                timestamps.append(w["start"])
    return timestamps

def generate_video_clip(start_time, m3u8_url, clip_file, duration=15):
    """
    Cuts a video clip from the original m3u8 video stream.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_time),   # start time in seconds
        "-i", m3u8_url,           # input video
        "-t", str(duration),      # duration
        "-c", "copy",             # fast cut (might shift to nearest keyframe)
        clip_file
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return clip_file 
  
target_word = st.text_input("Enter word to clip", "")

if st.button("Generate Video Clips from Word"):

    try:
        with st.status(f"Generating clips for '{target_word}'...", expanded=True) as status:
            if target_word and st.session_state.result_en:
                timestamps = find_all_word_timestamps(
                    st.session_state.result_en, target_word
                )

                print(f"Found {len(timestamps)} occurrence(s) of '{target_word}'")
                
                if timestamps:
                    st.success(f"Found {len(timestamps)} occurrence(s) of '{target_word}'")
                    
                    for i, ts in enumerate(timestamps, start=1):
                        clip_file = f"clip_{target_word}_{i}.mp4"
                        generate_video_clip(ts, st.session_state.m3u8_url, clip_file, duration=15)
                        
                        st.write(f"🎬 Clip {i} (from {ts:.2f}s → {ts+15:.2f}s)")
                        st.video(clip_file)
                else:
                    st.error(f"❌ Word '{target_word}' not found in transcript.") 
        status.update(label="Generated Clips", state="complete")

    except Exception as e:
        st.error(f"Error generating clips: {str(e)}")            




