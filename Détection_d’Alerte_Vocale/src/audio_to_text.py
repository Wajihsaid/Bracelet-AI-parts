from vosk import Model, KaldiRecognizer
import wave
import json
import os

def audio_to_text_vosk(audio_path):
    # Calculate correct model path relative to this script's location
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(root_dir, 'vosk_model', 'vosk-model-small-en-us-0.15')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Vosk model not found at: {model_path}")
    
    model = Model(model_path)

    wf = wave.open(audio_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        raise ValueError("Audio file must be WAV format mono PCM.")

    rec = KaldiRecognizer(model, wf.getframerate())

    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            results.append(result.get("text", ""))
    final_result = json.loads(rec.FinalResult())
    results.append(final_result.get("text", ""))

    return " ".join(results)
