from faster_whisper import WhisperModel

def transcribe_audio(audio_path):
    model = WhisperModel("large-v2", device="cuda")
    segments, info = model.transcribe(audio_path, beam_size=5)
    
    print("감지된 언어:", info.language)
    print("변환된 텍스트:")
    for segment in segments:
        print(f"{segment.text}", end="")
        
    full_text = " ".join([segment.text for segment in segments])
    return full_text

audio_file = "mix.wav"
transcribed_text = transcribe_audio(audio_file)
