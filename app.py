import streamlit as st
import subprocess
import os
import whisper
import asyncio
import time
from deep_translator import GoogleTranslator

async def async_download_m3u8(m3u8_url, output_path="output.mp3"):
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

if st.button("Generate Transcript"):
    if m3u8_url:
        st.write("Downloading Audio...")

        # Convert the .m3u8 stream to .mp4
        output_file = "converted_audio.mp3"
        start_time = time.perf_counter()
    # result = download_m3u8(m3u8_url,output_file)
        result = asyncio.run(async_download_m3u8(m3u8_url))
        end_time = time.perf_counter()
    
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.6f} seconds")

        model = whisper.load_model("medium")
        st.write("Generating Transcript...")
        result = model.transcribe(output_file, verbose=True)
        
        st.subheader("Extracted Transcript")
        with open("transcript.txt", "w", encoding="utf-8") as file:
            file.write(result["text"])
        st.write(result["text"])
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
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    text_chunks = split_text(text)

    
    # Translate text to English
    translator = GoogleTranslator(source=source_lang, target="en")
    translated_chunks = [translator.translate(chunk) for chunk in text_chunks] 
    print(translated_chunks)
    transated_text  = " ".join(translated_chunks)
    return transated_text

if st.button("Translate To English"):
    st.write(translate_to_english("transcript.txt"))




