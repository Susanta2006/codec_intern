# codec_intern
Machine Learning & NLP Projects Portfolio

This repository contains two projects demonstrating skills in Deep Learning and Natural Language Processing (NLP).

# Project 1: Speech-to-Text Transcription

A Python-based tool that converts spoken language into written text using the SpeechRecognition library and Google Web Speech API.

**Features**

    Live Transcription: Capture and process audio directly from the microphone.

    File Processing: Transcribe pre-recorded .wav audio files.

    Noise Calibration: Automatically adjusts for background noise to improve accuracy.

    Error Handling: Robust handling for unintelligible speech or connectivity issues.

**Usage**

    python speech_to_text.py


# Project 2: Handwritten Digit Recognizer

A Convolutional Neural Network (CNN) trained to recognize handwritten digits (0-9) using the MNIST dataset.

**Features**

    Architecture: Implements a CNN with 3 Convolutional layers using TensorFlow/Keras.

    Pre-processing: Automated image reshaping and normalization.

    Visualization: Matplotlib integration to show random test images with their predicted vs. true labels.

    Persistence: Saves the trained model as mnist_model.h5 for instant loading.

**Usage**

    python digit_recognizer.py


## Requirements

    To run these projects, install the following dependencies:

     pip install numpy matplotlib tensorflow SpeechRecognition PyAudio


Note: PyAudio is required specifically for microphone input.

Author

[Your Name]
