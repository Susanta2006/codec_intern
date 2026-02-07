import speech_recognition as sr
import os

def transcribe_from_mic():
    """
    Captures audio from the default microphone and transcribes it to text.
    """
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("\n--- Live Microphone Transcription ---")
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening... (Speak now)")
        
        try:
            # timeout: seconds it will wait for speech to start
            # phrase_time_limit: seconds it will record after speech starts
            audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            print("Processing audio...")
            
            # Using Google Web Speech API (Free, requires internet)
            text = recognizer.recognize_google(audio_data)
            return text
            
        except sr.WaitTimeoutError:
            return "Error: Listening timed out while waiting for phrase."
        except sr.UnknownValueError:
            return "Error: Speech was unintelligible."
        except sr.RequestError as e:
            return f"Error: Could not request results from service; {e}"

def transcribe_file(file_path):
    """
    Transcribes an existing .wav audio file.
    """
    recognizer = sr.Recognizer()
    
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' not found."

    with sr.AudioFile(file_path) as source:
        print(f"\n--- Processing File: {file_path} ---")
        audio_data = recognizer.record(source)
        
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Error: Could not understand audio in file."
        except sr.RequestError as e:
            return f"Error: Service request failed; {e}"

def main():
    print("Python Speech-to-Text Tool")
    print("==========================")
    print("1. Transcribe from Microphone (Live)")
    print("2. Transcribe from .wav File")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == '1':
        result = transcribe_from_mic()
        print(f"\nFinal Result:\n{result}")
    elif choice == '2':
        path = input("Enter the path to your .wav file: ").strip()
        result = transcribe_file(path)
        print(f"\nFinal Result:\n{result}")
    else:
        print("Invalid choice. Please restart the script.")

if __name__ == "__main__":
    # Note: To use this, you must install dependencies:
    # pip install SpeechRecognition PyAudio
    main()
