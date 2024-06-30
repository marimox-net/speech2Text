import os
import time
import numpy as np
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from threading import Thread
from faster_whisper import WhisperModel
import wave
import pyaudio
import concurrent.futures

HALLUCINATION_TEXTS = [
    "ご視聴ありがとうございました", "ご視聴ありがとうございました。",
    "ありがとうございました", "ありがとうございました。",
    "どうもありがとうございました", "どうもありがとうございました。",
    "どうも、ありがとうございました", "どうも、ありがとうございました。",
    "おやすみなさい", "おやすみなさい。",
    "Thanks for watching!",
    "終わり", "おわり",
    "お疲れ様でした", "お疲れ様でした。",
]

# モデルのロード
MODEL_SIZE = "large-v3"
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")

# パラメータ設定
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
SILENCE_THRESHOLD = 10  # 無音判定のしきい値
SILENCE_DURATION = 0.2  # 無音判定する持続時間 (秒)
OUT_DURATION = 0.5 # 強制的に途中出力する時間(秒)
MIN_AUDIO_LENGTH = 0.1  # 最小音声長 (秒)

# PyAudioインスタンス作成
audio = pyaudio.PyAudio()

# ストリームの設定
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

def record_audio(audio_directory):
    frames = []
    recording = False
    silent_chunks = 0
    speak_chunks = 0
    speak_cnt = 1

    audio_directory.mkdir(parents=True, exist_ok=True)

    def is_silent(data):
        # 無音かどうかを判定する関数
        rms = np.sqrt(np.mean(np.square(np.frombuffer(data, dtype=np.int16))))
        return rms < SILENCE_THRESHOLD

    def save_wave_file(filename, frames):
        # 録音データをWAVファイルとして保存する関数
        with wave.open(str(filename), 'wb') as wf:  # Pathオブジェクトを文字列に変換
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

    while True:
        data = stream.read(CHUNK)
        silent = is_silent(data)

        if silent:
            silent_chunks += 1
            speak_chunks = 0
        else:
            silent_chunks = 0
            speak_chunks += 1

        if silent_chunks > (SILENCE_DURATION * RATE / CHUNK):
            if recording:
                if len(frames) * CHUNK / RATE < MIN_AUDIO_LENGTH:
                    return
                else:
                    # 無音状態が続いたら録音を停止してファイルを保存
                    file_path = audio_directory / f"recorded_audio_{file_name}_latest.wav"
                    executor.submit(save_wave_file, Path(file_path), frames)
                frames = []
                recording = False
                speak_cnt = 1
        else:
            if not recording:
                file_name = f"{int(time.time())}"
                recording = True

            if speak_chunks > (OUT_DURATION * RATE / CHUNK):
                file_path = audio_directory / f"recorded_audio_{file_name}_{speak_cnt}.wav"
                executor.submit(save_wave_file, Path(file_path), frames)
                speak_cnt += 1
                speak_chunks = 0

            frames.append(data)

class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        
        file_name, file_ext = os.path.splitext(event.src_path)
        if not file_name.endswith("_latest"):
            base_name = file_name.rsplit('_', 1)[0]
            suffix = int(file_name.rsplit('_', 1)[-1])
            if os.path.exists(os.path.join(os.path.dirname(event.src_path), f"{base_name}_latest.wav")):
                # 最終ファイルがあるので処理不要、ファイル削除してスキップ
                os.remove(event.src_path)
                return
            if os.path.exists(os.path.join(os.path.dirname(event.src_path), f"{base_name}_{suffix + 1}.wav")):
                # 次ファイルがあるので処理不要、ファイル削除してスキップ
                os.remove(event.src_path)
                return

        # 文字起こしして、ファイルを削除
        self.process_file(event.src_path)
        os.remove(event.src_path)

    def process_file(self, file_path):
        # 文字起こし
        transcription = self.transcribe(file_path)

        # ハルシネーションで出力された可能性のある場合は処理しない
        if transcription in HALLUCINATION_TEXTS:
            return
        
        if transcription:
            if "latest" in str(file_path):
                # 最終ファイルの場合、そのまま出力
                print(transcription)
            else:
                # 喋っている途中の文字起こしは《》で囲う
                print("《"+transcription+"》")
    
    def transcribe(self, file_path):
        try:
            with open(file_path, 'rb') as audio_file:
                segments, _ = model.transcribe(audio_file, language="ja", beam_size=5, patience=0.5)
                transcription = ''.join(segment.text for segment in segments)
            return transcription
        except Exception as e:
            print(f"Error in transcribe: {e}")
            return ""

def start_monitoring(watch_path):
    # 録音処理(スレッドを立てる)
    # wavファイルを生成し続ける処理
    record_thread = Thread(target=record_audio, args=(watch_path,))
    record_thread.daemon = True
    record_thread.start()

    # フォルダを監視してwavファイルが生成された場合
    # 文字起こし処理を行う
    event_handler = FileHandler()
    observer = Observer()
    observer.schedule(event_handler, watch_path, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_monitoring(Path.cwd() / "tmp")
