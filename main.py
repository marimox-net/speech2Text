import os
import time
import numpy as np
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from threading import Thread
import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel

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

def record_audio(audio_directory, fs=16000, silence_threshold=0.3, min_duration=0.1, amplitude_threshold=0.01, out_duration=0.5):
    # wavファイルの出力先を作成
    audio_directory.mkdir(parents=True, exist_ok=True)

    # 録音処理
    while True:
        file_name = f"recorded_audio_{int(time.time())}"
        recorded_audio = []
        silent_time = 0
        speak_time = 0
        speak_cnt = 1

        try:
            with sd.InputStream(samplerate=fs, channels=1) as stream:
                # 最初に無音状態が終わるまでは録音せずに待機
                while True:
                    data, overflowed = stream.read(int(fs * min_duration))
                    if overflowed:
                        print("Overflow occurred. Some samples might have been lost.")
                    if np.any(np.abs(data) >= amplitude_threshold):
                        recorded_audio.append(data)
                        break
                
                # 録音を開始してから無音状態になるまでループ
                while True:
                    data, overflowed = stream.read(int(fs * min_duration))
                    if overflowed:
                        print("Overflow occurred. Some samples might have been lost.")
                    recorded_audio.append(data)
                    if np.all(np.abs(data) < amplitude_threshold):
                        silent_time += min_duration
                        if silent_time >= silence_threshold:
                            break
                    else:
                        # 一定時間経過で話し途中でもwavファイル作成
                        speak_time += min_duration
                        if speak_time >= out_duration:
                            file_path = audio_directory / f"{file_name}_{speak_cnt}.wav"
                            speak_time = 0
                            speak_cnt += 1
                            audio_data = np.concatenate(recorded_audio, axis=0)
                            audio_data = np.int16(audio_data * 32767)
                            write(file_path, fs, audio_data)
                        silent_time = 0
        except Exception as e:
            print(f"Error in record_audio: {e}")
            continue
        
        # 無音検知によるwavファイル作成
        file_path = audio_directory / f"{file_name}_latest.wav"
        audio_data = np.concatenate(recorded_audio, axis=0)
        audio_data = np.int16(audio_data * 32767)
        write(file_path, fs, audio_data)

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
    record_thread = Thread(target=record_audio, args=(watch_path, 16000, 0.3, 0.1, 0.01, 0.3))
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
