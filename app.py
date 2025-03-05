import streamlit as st
import subprocess
import os
import whisper
import asyncio
import time
from deep_translator import GoogleTranslator
import requests
from urllib.parse import urlparse, parse_qs

output_path = "output.mp3"
output_transcript_file_path = "transcript.txt"

# Load the Whisper model only once
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("medium")

# initilaise it only once 
model = load_whisper_model()
    
async def async_download_m3u8(m3u8_url):
    """Downloads and converts an .m3u8 stream to .mp3 using ffmpeg asynchronously."""
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-y", 
        "-i", m3u8_url,
        "-q:a", "0",
        "-map", "a",
        output_path,
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

def format_timestamp(seconds):
    millis = int((seconds % 1) * 1000)
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02}:{seconds:02}.{millis:03}"

# Read the transcript file
def read_transcript(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()
    
if st.button("Generate Transcript"):
    if m3u8_url:
        with st.status("Downloading Audio...", expanded=True) as status:
            result = asyncio.run(async_download_m3u8(m3u8_url))
            status.update(label="Audio Downloaded", state="complete")

        with st.status("Generating Transcript...", expanded=True) as status:
           result = model.transcribe(output_path, verbose=True)
           status.update(label="Transcript Generated",state="complete")

        with open(output_transcript_file_path, "w", encoding="utf-8") as file:
            for segment in result["segments"]:
                start_time = format_timestamp(segment["start"])
                end_time = format_timestamp(segment["end"])
                text = segment["text"].strip()
                subtitle_line = f"[{start_time} --> {end_time}]  {text}\n"
                file.write(subtitle_line)
        
        with st.status("Showing Transcript", expanded=True) as status:
            st.text_area("Transcript", read_transcript(output_transcript_file_path), height= 500)
            status.update(label="Transcript Displayed",state="complete")

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

    with st.status("Translating in English", expanded = True) as status:
        translator = GoogleTranslator(source=source_lang, target="en")
        translated_chunks = [translator.translate(chunk) for chunk in text_chunks] 
        translated_text  = " ".join(translated_chunks)
        st.text_area("English Translation", translated_text, height = 500)
        status.update(label="Translated in English",state="complete")



if st.button("Translate To English"):
    translate_to_english(output_transcript_file_path)







