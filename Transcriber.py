import pyaudio
import wave
import keyboard
import whisper
import warnings


warnings.simplefilter("ignore")

def start_recording(output_filename):
    chunk_size = 1024
    format = pyaudio.paInt16
    channels = 1
    frame_rate = 44100


    p = pyaudio.PyAudio()

    stream = p.open(format=format,
                    channels=channels,
                    rate=frame_rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    print("* Recording started")

    frames = []

    while True:
        data = stream.read(chunk_size)
        frames.append(data)

        if keyboard.is_pressed('q'):
            break

    print("* Recording stopped")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(frame_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

def main():

    model = whisper.load_model("base.en")
    
    output_filename = "recorded_audio.wav"
    while True:
        print("Press 's' to start recording and 'q' to stop recording:")
        keyboard.wait('s')
        start_recording(output_filename)

        print("Generating Transcript.......")
        transcription = model.transcribe(audio=output_filename)
        
        print("Transcript : " + transcription['text'] )

if __name__ == "__main__":
 
    main()


