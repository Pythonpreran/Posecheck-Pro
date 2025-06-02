# PoseCheck Pro: AI-Powered Exercise Trainer

![ezgif com-video-to-gif-converter (1) (1) (1)](https://github.com/user-attachments/assets/541803ee-6709-4f82-925e-b06d604992aa)


PoseCheck Pro is an AI-powered web app that guides you through home workouts with real-time posture correction, audio instructions, and personalized stats. Built for a hackathon, it ensures safe exercise form using computer vision, helping prevent injuries from improper squats or arm raises. Whether you're a fitness enthusiast or a developer exploring health tech, PoseCheck Pro offers a seamless, webcam-based fitness experience.

## Features

- **Real-Time Posture Correction**: Uses Mediapipe to track pose landmarks, providing instant feedback (e.g., "Level your shoulders") for arm raises (150-210° angles) and squats (50-90° knee angle).
- **Audio Guidance**: Delivers step-by-step instructions (e.g., "Hold arms up for 3 seconds") via gTTS, cached as MP3s, played without a visible play bar using JavaScript.
- **Personalized Stats**: Tracks reps, exercise duration (starts when camera activates), and calories burned (MET-based) based on user height and weight.
- **Intuitive UI**: Streamlit-powered interface with a two-column layout (webcam feed/debug on left, feedback on right) and sidebar for exercise selection (Arm Raise, Squat) and reps.
- **Webcam-Only**: Runs on any device with a webcam, with calibration tips for optimal tracking (e.g., "Stand 3-6 feet from camera").

## Screenshots
![vlcsnap-2025-06-02-23h04m50s369](https://github.com/user-attachments/assets/fa905e56-cac6-4de0-91dd-754ff8503df7)
![vlcsnap-2025-06-02-23h04m59s335](https://github.com/user-attachments/assets/21354c9a-5ec8-446d-ac53-9ddf64e11b14)

## Technologies

- **Python**: Core language for app logic.
- **Streamlit**: Web framework for the UI.
- **Mediapipe**: Pose estimation for landmark tracking.
- **OpenCV**: Webcam video capture and processing.
- **gTTS**: Text-to-speech for audio instructions.
- **pydub**: Audio processing and MP3 caching.
- **FFmpeg**: Backend for pydub audio operations.
- **JavaScript**: Audio playback via Streamlit components.
- **NumPy**: Coordinate and angle calculations.
- **Math**: Angle computations (atan2, degrees).

## Prerequisites

- Python 3.8+
- A webcam (built-in or external)
- FFmpeg installed and added to system PATH
- Git (for cloning the repository)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Pythonpreran/Posecheck-Pro.git
   cd Posecheck-Pro
   ```

2. **Install Python Dependencies**:
   ```bash
   pip install streamlit opencv-python mediapipe gTTS pydub
   ```

3. **Install FFmpeg**:
   - **Windows**:
     1. Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) or a trusted source (e.g., [gyan.dev](https://www.gyan.dev/ffmpeg/builds/)).
     2. Extract the archive (e.g., to `C:\ffmpeg`).
     3. Add `C:\ffmpeg\bin` to your system PATH:
        - Right-click 'This PC' > Properties > Advanced system settings > Environment Variables.
        - Edit 'Path' under System variables, add `C:\ffmpeg\bin`, and click OK.
     4. Verify: Open Command Prompt and run `ffmpeg -version`.
   - **macOS**:
     ```bash
     brew install ffmpeg
     ```
   - **Linux**:
     ```bash
     sudo apt-get install ffmpeg
     ```

4. **Verify Setup**:
   - Ensure a webcam is connected.
   - Check Python version: `python --version` (should be 3.8+).
   - Confirm FFmpeg: `ffmpeg -version`.

## Usage

1. **Run the App**:
   ```bash
   streamlit run main.py
   ```

2. **Interact with the App**:
   - Open the provided URL (e.g., `http://localhost:8501`) in a browser.
   - In the sidebar:
     - Select an exercise (Arm Raise or Squat).
     - Set number of repetitions (1-20).
     - Enter height (100-250 cm) and weight (30-200 kg).
   - Click **Start Exercise** to begin.
   - Follow calibration tips (good lighting, 3-6 feet from camera, full body visible).
   - Watch the webcam feed for pose landmarks and listen to audio instructions.
   - View real-time feedback and workout summary (reps, time, calories) upon completion.

3. **Reset or Stop**:
   - Click **Reset Exercise** in the sidebar to restart.
   - Close the terminal to stop the app.

## Project Structure

```
Posecheck-Pro/
├── main.py             # Main Streamlit app
├── audio_cache/        # Directory for cached MP3 audio files (auto-generated)
├── README.md           # This file
└── requirements.txt    # Optional: List of dependencies (recommended)
```

## Troubleshooting

- **Webcam Not Found**: Ensure a webcam is connected and accessible. Check `cv2.VideoCapture(0)`; try `1` or `2` if multiple cameras exist.
- **FFmpeg Error**: Verify FFmpeg is in PATH (`ffmpeg -version`). Reinstall if needed.
- **Mediapipe Warnings**: Ensure frame dimensions are set (640x480 in code). Update Mediapipe if errors persist (`pip install --upgrade mediapipe`).
- **Audio Not Playing**: Check browser autoplay settings. Ensure `audio_cache` directory is writable.
- **Performance Lag**: Reduce webcam resolution or use a lower `model_complexity` in Mediapipe (e.g., 0 instead of 1).

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a branch: `git checkout -b feature/your-feature`.
3. Commit changes: `git commit -m "Add your feature"`.
4. Push to the branch: `git push origin feature/your-feature`.
5. Open a pull request with a clear description.

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md) and report issues via [GitHub Issues](https://github.com/Pythonpreran/Posecheck-Pro/issues).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Built for the hackathon Neural Nexus to explore health tech and AI.
- Inspired by the need for safe home workouts.
- Thanks to Mediapipe, Streamlit, and gTTS communities for amazing tools.

## Contact

For questions or feedback, reach out via [GitHub Issues](https://github.com/Pythonpreran/Posecheck-Pro/issues) or connect on [LinkedIn](https://www.linkedin.com/in/your-profile). <!-- Replace with your LinkedIn -->
