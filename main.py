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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

hallucinationTexts = [
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
model_size = "large-v3"
model = WhisperModel(model_size, device="cuda", compute_type="float16")

def recordAudio(audioDirectory, fs=16000, silenceThreshold=0.3, minDuration=0.1, amplitudeThreshold=0.01, outDuration=0.5):
    # wavファイルの出力先を作成
    audioDirectory.mkdir(parents=True, exist_ok=True)

    # 録音処理
    while True:
        fileName = f"recorded_audio_{int(time.time())}"
        recordedAudio = []
        silentTime = 0
        speakTime = 0
        speakCnt = 1

        try:
            with sd.InputStream(samplerate=fs, channels=1) as stream:
                # 最初に無音状態が終わるまでは録音せずに待機
                while True:
                    data, overflowed = stream.read(int(fs * minDuration))
                    if overflowed:
                        print("Overflow occurred. Some samples might have been lost.")
                    if np.any(np.abs(data) >= amplitudeThreshold):
                        recordedAudio.append(data)
                        break
                
                # 録音を開始してから無音状態になるまでループ
                while True:
                    data, overflowed = stream.read(int(fs * minDuration))
                    if overflowed:
                        print("Overflow occurred. Some samples might have been lost.")
                    recordedAudio.append(data)
                    if np.all(np.abs(data) < amplitudeThreshold):
                        silentTime += minDuration
                        if silentTime >= silenceThreshold:
                            break
                    else:
                        # 一定時間経過で話し途中でもwavファイル作成
                        speakTime += minDuration
                        if speakTime >= outDuration:
                            filePath = audioDirectory / f"{fileName}_{speakCnt}.wav"
                            speakTime = 0
                            speakCnt += 1
                            audioData = np.concatenate(recordedAudio, axis=0)
                            audioData = np.int16(audioData * 32767)
                            write(filePath, fs, audioData)
                        silentTime = 0
        except Exception as e:
            print(f"Error in recordAudio: {e}")
            continue
        
        # 無音検知によるwavファイル作成
        filePath = audioDirectory / f"{fileName}_latest.wav"
        audioData = np.concatenate(recordedAudio, axis=0)
        audioData = np.int16(audioData * 32767)
        write(filePath, fs, audioData)

class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        
        file_name, file_ext = os.path.splitext(event.src_path)
        if not file_name.endswith("_latest"):
            baseName = file_name.rsplit('_', 1)[0]
            suffix = int(file_name.rsplit('_', 1)[-1])
            if os.path.exists(os.path.join(os.path.dirname(event.src_path), f"{baseName}_latest.wav")):
                # 最終ファイルがあるので処理不要、ファイル削除してスキップ
                os.remove(event.src_path)
                return
            if os.path.exists(os.path.join(os.path.dirname(event.src_path), f"{baseName}_{suffix+1}.wav")):
                # 次ファイルがあるので処理不要、ファイル削除してスキップ
                os.remove(event.src_path)
                return

        # 文字起こしして、ファイルを削除
        self.process_file(event.src_path)
        os.remove(event.src_path)

    def process_file(self, filePath):
        # 文字起こし
        transcription = self.transcribe(filePath)

        # ハルシネーションで出力された可能性のある場合は処理しない
        if transcription in hallucinationTexts:
            return
        
        if transcription:
            if "latest" in str(filePath):
                # 最終ファイルの場合、そのまま出力
                print(transcription)
            else:
                # 喋っている途中の文字起こしは《》で囲う
                print("《"+transcription+"》")
    
    def transcribe(self, filePath):
        try:
            with open(filePath, 'rb') as audioFile:
                segments, _ = model.transcribe(audioFile, language="ja", beam_size=5,patience=0.5)
                transcription = ''.join(segment.text for segment in segments)
            return transcription
        except Exception as e:
            print(f"Error in transcribe: {e}")
            return ""

def startMonitoring(watchPath):
    # 録音処理(スレッドを立てる)
    # wavファイルを生成し続ける処理
    recordThread = Thread(target=recordAudio, args=(watchPath, 16000, 0.3, 0.1, 0.01, 0.3))
    recordThread.daemon = True
    recordThread.start()

    # フォルダを監視してwavファイルが生成された場合
    # 文字起こし処理を行う
    eventHandler = FileHandler()
    observer = Observer()
    observer.schedule(eventHandler, watchPath, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    startMonitoring(Path.cwd() / "tmp")