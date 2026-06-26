pip install faster-whisper sounddevice soundfile numpy ctranslate2
Install FFmpeg and add it to PATH.
Run: python offline_speech_to_text.py
The script will automatically detect supported audio files in the current folder, transcribe them, and save a .txt file next to each audio file.
