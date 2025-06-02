import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from math import atan2, degrees
import time
from gtts import gTTS
import base64
import io
import os
from pydub import AudioSegment
import streamlit.components.v1 as components

# Mediapipe Pose setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

st.set_page_config(layout="wide")
st.title("🧘‍♂️ PoseCheck Pro - Guided AI Exercise Trainer")

# Ensure audio cache directory exists
AUDIO_CACHE_DIR = os.path.join(os.path.dirname(__file__), "audio_cache")
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

# Get landmark coordinates
def get_coords(landmarks, name):
    lm = mp_pose.PoseLandmark[name]
    return [landmarks[lm].x, landmarks[lm].y, landmarks[lm].visibility]

# Compute angle between 3 points
def get_angle(a, b, c):
    ang = degrees(atan2(c[1]-b[1], c[0]-b[0]) - atan2(a[1]-b[1], a[0]-b[0]))
    return abs(ang) if ang >= 0 else 360 + ang

# Text-to-speech for instructions with file caching
def text_to_speech(text, filename):
    file_path = os.path.join(AUDIO_CACHE_DIR, filename)
    if os.path.exists(file_path):
        audio = AudioSegment.from_file(file_path, format="mp3")
        with open(file_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode()
        return audio_base64, audio.duration_seconds
    else:
        tts = gTTS(text=text, lang='en')
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        audio = AudioSegment.from_file(buffer, format="mp3")
        audio.export(file_path, format="mp3")
        audio_base64 = base64.b64encode(buffer.getvalue()).decode()
        return audio_base64, audio.duration_seconds

# Play audio in backend using JavaScript
def play_audio_backend(audio_base64, key):
    audio_data = f"data:audio/mp3;base64,{audio_base64}"
    js_code = f"""
    <script>
        var audio = new Audio('{audio_data}');
        audio.play().catch(function(error) {{
            console.log("Audio playback error: ", error);
        }});
    </script>
    """
    components.html(js_code, height=0, width=0)

# Feedback for improper form
def get_arm_raise_feedback(landmarks, step_idx):
    feedback = []
    if step_idx == 0:  # Standing posture
        l_shoulder = get_coords(landmarks, 'LEFT_SHOULDER')
        r_shoulder = get_coords(landmarks, 'RIGHT_SHOULDER')
        l_hip = get_coords(landmarks, 'LEFT_HIP')
        r_hip = get_coords(landmarks, 'RIGHT_HIP')
        l_wrist = get_coords(landmarks, 'LEFT_WRIST')
        r_wrist = get_coords(landmarks, 'RIGHT_WRIST')
        if abs(l_shoulder[1] - r_shoulder[1]) >= 0.1:
            feedback.append("Level your shoulders.")
        if abs(l_shoulder[1] - l_hip[1]) <= 0.15 or abs(r_shoulder[1] - r_hip[1]) <= 0.15:
            feedback.append("Stand upright with a straight torso.")
        if l_wrist[1] <= l_shoulder[1] or r_wrist[1] <= r_shoulder[1]:
            feedback.append("Relax your arms by your sides.")
        if min(l_shoulder[2], r_shoulder[2], l_hip[2], r_hip[2], l_wrist[2], r_wrist[2]) <= 0.1:
            feedback.append("Ensure your body is fully visible in the camera.")
    elif step_idx == 1 or step_idx == 4:  # Arms horizontal
        l_shoulder = get_coords(landmarks, 'LEFT_SHOULDER')
        r_shoulder = get_coords(landmarks, 'RIGHT_SHOULDER')
        l_elbow = get_coords(landmarks, 'LEFT_ELBOW')
        r_elbow = get_coords(landmarks, 'RIGHT_ELBOW')
        l_wrist = get_coords(landmarks, 'LEFT_WRIST')
        r_wrist = get_coords(landmarks, 'RIGHT_WRIST')
        l_angle = get_angle(l_shoulder, l_elbow, l_wrist)
        r_angle = get_angle(r_shoulder, r_elbow, r_wrist)
        if abs(l_angle - 180) > 30 or abs(r_angle - 180) > 30:
            feedback.append("Extend both arms horizontally to form a straight line.")
        if abs(l_wrist[1] - l_shoulder[1]) > 0.1 or abs(r_wrist[1] - r_shoulder[1]) > 0.1:
            feedback.append("Align your wrists with your shoulders horizontally.")
        if min(l_shoulder[2], l_elbow[2], l_wrist[2], r_shoulder[2], r_elbow[2], r_wrist[2]) <= 0.1:
            feedback.append("Ensure both arms are fully visible.")
    elif step_idx == 2 or step_idx == 3:  # Arms straight up
        l_shoulder = get_coords(landmarks, 'LEFT_SHOULDER')
        r_shoulder = get_coords(landmarks, 'RIGHT_SHOULDER')
        l_elbow = get_coords(landmarks, 'LEFT_ELBOW')
        r_elbow = get_coords(landmarks, 'RIGHT_ELBOW')
        l_wrist = get_coords(landmarks, 'LEFT_WRIST')
        r_wrist = get_coords(landmarks, 'RIGHT_WRIST')
        l_angle = get_angle(l_shoulder, l_elbow, l_wrist)
        r_angle = get_angle(r_shoulder, r_elbow, r_wrist)
        if abs(l_angle - 180) > 30 or abs(r_angle - 180) > 30:
            feedback.append("Straighten both arms vertically.")
        if l_wrist[1] >= l_shoulder[1] - 0.1 or r_wrist[1] >= r_shoulder[1] - 0.1:
            feedback.append("Raise both wrists higher, above shoulder level.")
        if min(l_shoulder[2], l_elbow[2], l_wrist[2], r_shoulder[2], r_elbow[2], r_wrist[2]) <= 0.1:
            feedback.append("Ensure both arms are fully visible.")
    elif step_idx == 5:  # Arms lowered
        l_wrist = get_coords(landmarks, 'LEFT_WRIST')
        r_wrist = get_coords(landmarks, 'RIGHT_WRIST')
        l_hip = get_coords(landmarks, 'LEFT_HIP')
        r_hip = get_coords(landmarks, 'RIGHT_HIP')
        if l_wrist[1] <= l_hip[1] or r_wrist[1] <= r_hip[1]:
            feedback.append("Lower both arms below hip level.")
        if min(l_wrist[2], r_wrist[2], l_hip[2], r_hip[2]) <= 0.1:
            feedback.append("Ensure both arms are fully visible.")
    return feedback

def get_squat_feedback(landmarks, step_idx):
    feedback = []
    hip = get_coords(landmarks, 'RIGHT_HIP')
    knee = get_coords(landmarks, 'RIGHT_KNEE')
    ankle = get_coords(landmarks, 'RIGHT_ANKLE')
    angle = get_angle(hip, knee, ankle)
    if step_idx == 0 or step_idx == 2:  # Standing
        if angle <= 140:
            feedback.append("Straighten your knees to a knee angle greater than 140°.")
        if min(hip[2], knee[2], ankle[2]) <= 0.1:
            feedback.append("Ensure your legs are fully visible.")
    elif step_idx == 1:  # Squat down
        if angle <= 50 or angle >= 90:
            feedback.append("Bend your knees to a 50-90° angle.")
        if hip[1] <= knee[1]:
            feedback.append("Lower your hips below knee level.")
        if min(hip[2], knee[2], ankle[2]) <= 0.1:
            feedback.append("Ensure your legs are fully visible.")
    return feedback

# Validators for exercise 1 - Arm Raise
def check_standing_posture(landmarks):
    l_shoulder = get_coords(landmarks, 'LEFT_SHOULDER')
    r_shoulder = get_coords(landmarks, 'RIGHT_SHOULDER')
    l_hip = get_coords(landmarks, 'LEFT_HIP')
    r_hip = get_coords(landmarks, 'RIGHT_HIP')
    l_wrist = get_coords(landmarks, 'LEFT_WRIST')
    r_wrist = get_coords(landmarks, 'RIGHT_WRIST')
    shoulder_level = abs(l_shoulder[1] - r_shoulder[1]) < 0.1
    torso_upright = abs(l_shoulder[1] - l_hip[1]) > 0.15 and abs(r_shoulder[1] - r_hip[1]) > 0.15
    arms_relaxed = l_wrist[1] > l_shoulder[1] and r_wrist[1] > r_shoulder[1]
    visibility = min(l_shoulder[2], r_shoulder[2], l_hip[2], r_hip[2], l_wrist[2], r_wrist[2]) > 0.1
    debug_container.write(f"**Debug Info for Step 1**")
    debug_container.write(f"Shoulder Level: {abs(l_shoulder[1] - r_shoulder[1]):.3f} (< 0.1)")
    debug_container.write(f"Torso Upright: L={abs(l_shoulder[1] - l_hip[1]):.3f}, R={abs(r_shoulder[1] - r_hip[1]):.3f} (> 0.15)")
    debug_container.write(f"Arms Relaxed: L_Wrist={l_wrist[1]:.3f} > L_Shoulder={l_shoulder[1]:.3f}, "
                         f"R_Wrist={r_wrist[1]:.3f} > R_Shoulder={r_shoulder[1]:.3f}")
    debug_container.write(f"Visibility: {min(l_shoulder[2], r_shoulder[2], l_hip[2], r_hip[2], l_wrist[2], r_wrist[2]):.3f} (> 0.1)")
    return shoulder_level and torso_upright and arms_relaxed and visibility

def check_arms_horizontal(landmarks):
    l_shoulder = get_coords(landmarks, 'LEFT_SHOULDER')
    r_shoulder = get_coords(landmarks, 'RIGHT_SHOULDER')
    l_elbow = get_coords(landmarks, 'LEFT_ELBOW')
    r_elbow = get_coords(landmarks, 'RIGHT_ELBOW')
    l_wrist = get_coords(landmarks, 'LEFT_WRIST')
    r_wrist = get_coords(landmarks, 'RIGHT_WRIST')
    l_angle = get_angle(l_shoulder, l_elbow, l_wrist)
    r_angle = get_angle(r_shoulder, r_elbow, r_wrist)
    wrist_height = abs(l_wrist[1] - l_shoulder[1]) < 0.1 and abs(r_wrist[1] - r_shoulder[1]) < 0.1
    visibility = min(l_shoulder[2], l_elbow[2], l_wrist[2], r_shoulder[2], r_elbow[2], r_wrist[2]) > 0.1
    return 150 < l_angle < 210 and 150 < r_angle < 210 and wrist_height and visibility

def check_arms_up(landmarks):
    l_shoulder = get_coords(landmarks, 'LEFT_SHOULDER')
    r_shoulder = get_coords(landmarks, 'RIGHT_SHOULDER')
    l_elbow = get_coords(landmarks, 'LEFT_ELBOW')
    r_elbow = get_coords(landmarks, 'RIGHT_ELBOW')
    l_wrist = get_coords(landmarks, 'LEFT_WRIST')
    r_wrist = get_coords(landmarks, 'RIGHT_WRIST')
    l_angle = get_angle(l_shoulder, l_elbow, l_wrist)
    r_angle = get_angle(r_shoulder, r_elbow, r_wrist)
    wrist_height = l_wrist[1] < l_shoulder[1] - 0.1 and r_wrist[1] < r_shoulder[1] - 0.1
    visibility = min(l_shoulder[2], l_elbow[2], l_wrist[2], r_shoulder[2], r_elbow[2], r_wrist[2]) > 0.1
    return 150 < l_angle < 210 and 150 < r_angle < 210 and wrist_height and visibility

def check_hold_stable(landmarks):
    global hold_start
    if hold_start is None:
        hold_start = time.time()
    elapsed = time.time() - hold_start
    l_wrist = get_coords(landmarks, 'LEFT_WRIST')
    r_wrist = get_coords(landmarks, 'RIGHT_WRIST')
    l_shoulder = get_coords(landmarks, 'LEFT_SHOULDER')
    r_shoulder = get_coords(landmarks, 'RIGHT_SHOULDER')
    if l_wrist[1] > l_shoulder[1] or r_wrist[1] > r_shoulder[1]:
        st.session_state.step = 0
        st.session_state.validation_count = 0
        hold_start = None
        return False
    return elapsed > 3 and check_arms_up(landmarks)

def check_arms_lowered(landmarks):
    l_wrist = get_coords(landmarks, 'LEFT_WRIST')
    r_wrist = get_coords(landmarks, 'RIGHT_WRIST')
    l_hip = get_coords(landmarks, 'LEFT_HIP')
    r_hip = get_coords(landmarks, 'RIGHT_HIP')
    visibility = min(l_wrist[2], r_wrist[2], l_hip[2], r_hip[2]) > 0.1
    return l_wrist[1] > l_hip[1] and r_wrist[1] > r_hip[1] and visibility

# Validators for exercise 2 - Squat
def check_squat_start(landmarks):
    hip = get_coords(landmarks, 'RIGHT_HIP')
    knee = get_coords(landmarks, 'RIGHT_KNEE')
    ankle = get_coords(landmarks, 'RIGHT_ANKLE')
    angle = get_angle(hip, knee, ankle)
    visibility = min(hip[2], knee[2], ankle[2]) > 0.1
    return angle > 140 and visibility

def check_squat_down(landmarks):
    hip = get_coords(landmarks, 'RIGHT_HIP')
    knee = get_coords(landmarks, 'RIGHT_KNEE')
    ankle = get_coords(landmarks, 'RIGHT_ANKLE')
    angle = get_angle(hip, knee, ankle)
    hip_height = hip[1] > knee[1]
    visibility = min(hip[2], knee[2], ankle[2]) > 0.1
    return 50 < angle < 90 and hip_height and visibility

def check_squat_up(landmarks):
    hip = get_coords(landmarks, 'RIGHT_HIP')
    knee = get_coords(landmarks, 'RIGHT_KNEE')
    ankle = get_coords(landmarks, 'RIGHT_ANKLE')
    angle = get_angle(hip, knee, ankle)
    visibility = min(hip[2], knee[2], ankle[2]) > 0.1
    return angle > 140 and visibility

# Define exercises
exercises = {
    "Arm Raise": [
        {"instruction": "Stand straight with arms relaxed", "validator": check_standing_posture, "feedback": get_arm_raise_feedback, "audio_file": "arm_raise_1.mp3"},
        {"instruction": "Raise both arms to horizontal", "validator": check_arms_horizontal, "feedback": get_arm_raise_feedback, "audio_file": "arm_raise_2.mp3"},
        {"instruction": "Raise both arms straight up", "validator": check_arms_up, "feedback": get_arm_raise_feedback, "audio_file": "arm_raise_3.mp3"},
        {"instruction": "Hold arms up for 3 seconds", "validator": check_hold_stable, "feedback": get_arm_raise_feedback, "audio_file": "arm_raise_4.mp3"},
        {"instruction": "Lower arms to horizontal", "validator": check_arms_horizontal, "feedback": get_arm_raise_feedback, "audio_file": "arm_raise_5.mp3"},
        {"instruction": "Lower arms to starting position", "validator": check_arms_lowered, "feedback": get_arm_raise_feedback, "audio_file": "arm_raise_6.mp3"},
    ],
    "Squat": [
        {"instruction": "Stand with feet shoulder-width apart", "validator": check_squat_start, "feedback": get_squat_feedback, "audio_file": "squat_1.mp3"},
        {"instruction": "Squat until thighs are parallel to ground", "validator": check_squat_down, "feedback": get_squat_feedback, "audio_file": "squat_2.mp3"},
        {"instruction": "Return to standing position", "validator": check_squat_up, "feedback": get_squat_feedback, "audio_file": "squat_3.mp3"},
    ]
}

# Pre-generate audio for all exercise instructions
audio_cache = {}
for exercise, steps in exercises.items():
    for step in steps:
        instruction = step['instruction']
        filename = step['audio_file']
        if instruction not in audio_cache:
            audio_base64, duration = text_to_speech(instruction, filename)
            audio_cache[instruction] = (audio_base64, duration)
audio_cache["completed"] = text_to_speech("Exercise Completed!", "completed.mp3")
calibration_text = """
Ensure good lighting, avoiding shadows or dim light.
Stand 3 to 6 feet from the camera.
Keep your full upper body and hips visible in the frame.
Face the camera directly with arms relaxed by your sides.
"""
audio_cache["calibration"] = text_to_speech(calibration_text, "calibration.mp3")

# Calculate calories burned (simplified MET-based formula)
def calculate_calories(weight_kg, duration_min, exercise):
    met = {"Arm Raise": 3.5, "Squat": 5.0}
    return 0.0175 * met[exercise] * weight_kg * duration_min

# Streamlit session state setup
if 'exercise' not in st.session_state:
    st.session_state.exercise = "Arm Raise"
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'completed' not in st.session_state:
    st.session_state.completed = False
if 'last_validated' not in st.session_state:
    st.session_state.last_validated = 0
if 'validation_count' not in st.session_state:
    st.session_state.validation_count = 0
if 'reps' not in st.session_state:
    st.session_state.reps = 1
if 'current_rep' not in st.session_state:
    st.session_state.current_rep = 0
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'height' not in st.session_state:
    st.session_state.height = 170
if 'weight' not in st.session_state:
    st.session_state.weight = 70
if 'started' not in st.session_state:
    st.session_state.started = False
if 'current_audio' not in st.session_state:
    st.session_state.current_audio = None
if 'audio_end_time' not in st.session_state:
    st.session_state.audio_end_time = 0

# Sidebar controls
st.sidebar.title("Exercise Selector")
st.session_state.exercise = st.sidebar.selectbox("Choose Exercise", list(exercises.keys()))
st.session_state.reps = st.sidebar.number_input("Number of Repetitions", min_value=1, max_value=20, value=1)
st.session_state.height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=170)
st.session_state.weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=70)

# Start and Reset buttons
all_info_provided = st.session_state.exercise and st.session_state.reps and st.session_state.height and st.session_state.weight
if st.sidebar.button("Start Exercise", disabled=not all_info_provided):
    st.session_state.started = True
    st.session_state.step = 0
    st.session_state.current_rep = 0
    st.session_state.completed = False
    st.session_state.validation_count = 0
    st.session_state.last_validated = 0
    st.session_state.current_audio = None
    st.session_state.audio_end_time = 0

if st.sidebar.button("Reset Exercise"):
    st.session_state.started = False
    st.session_state.step = 0
    st.session_state.current_rep = 0
    st.session_state.completed = False
    st.session_state.validation_count = 0
    st.session_state.last_validated = 0
    st.session_state.start_time = None
    st.session_state.current_audio = None
    st.session_state.audio_end_time = 0

# Calibration instructions with play button
st.sidebar.markdown("""
**Calibration Tips**:
- Ensure good lighting (avoid shadows or dim light).
- Stand 3-6 feet from the camera.
- Keep your full upper body and hips visible in the frame.
- Face the camera directly with arms relaxed by your sides.
""")
if st.sidebar.button("🔊 Play Calibration Tips"):
    audio_base64, duration = audio_cache["calibration"]
    play_audio_backend(audio_base64, key=f"calibration_{time.time()}")
    st.session_state.audio_end_time = time.time() + duration

# Layout
col1, col2 = st.columns([2, 1])
with col1:
    stframe = st.empty()
    debug_container = st.empty()
with col2:
    feedback_container = st.empty()

# Main loop
if not st.session_state.started:
    stframe.markdown("Please select an exercise, enter your height, weight, and number of repetitions, then click 'Start Exercise'.")
else:
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    hold_start = None
    VALIDATION_THRESHOLD = 5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam. Please check your camera connection.")
            break

        # Start timer when camera is active
        if st.session_state.start_time is None:
            st.session_state.start_time = time.time()

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        step_idx = st.session_state.step
        steps = exercises[st.session_state.exercise]

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark

            if step_idx < len(steps):
                step = steps[step_idx]
                # Play audio instruction if not currently playing and previous audio has finished
                current_time = time.time()
                if (st.session_state.current_audio != step['instruction'] and 
                    current_time >= st.session_state.audio_end_time):
                    audio_base64, duration = audio_cache[step['instruction']]
                    play_audio_backend(audio_base64, key=f"step_{step_idx}_rep_{st.session_state.current_rep}_{current_time}")
                    st.session_state.current_audio = step['instruction']
                    st.session_state.audio_end_time = current_time + duration

                cv2.putText(frame, f"Rep {st.session_state.current_rep + 1}/{st.session_state.reps} - Step {step_idx + 1}: {step['instruction']}", 
                           (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

                # Provide feedback
                feedback = step['feedback'](landmarks, step_idx)
                if feedback:
                    feedback_container.markdown("**Feedback:**\n" + "\n".join(f"- {f}" for f in feedback))
                else:
                    feedback_container.markdown("**Feedback:** Good form! Keep it up.")

                # Validate with debouncing and consecutive frame checking
                if current_time - st.session_state.last_validated > 0.3:
                    if step['validator'](landmarks):
                        st.session_state.validation_count += 1
                        if st.session_state.validation_count >= VALIDATION_THRESHOLD:
                            st.session_state.step += 1
                            st.session_state.validation_count = 0
                            st.session_state.last_validated = current_time
                            if step['instruction'].startswith("Hold"):
                                hold_start = None
                            if st.session_state.step >= len(steps):
                                st.session_state.current_rep += 1
                                st.session_state.step = 0
                                st.session_state.current_audio = None
                                if st.session_state.current_rep >= st.session_state.reps:
                                    st.session_state.completed = True
                    else:
                        st.session_state.validation_count = 0
                    st.session_state.last_validated = current_time
            else:
                st.session_state.step = 0
                st.session_state.current_audio = None

        if st.session_state.completed:
            duration = (time.time() - st.session_state.start_time) / 60
            calories = calculate_calories(st.session_state.weight, duration, st.session_state.exercise)
            cv2.putText(frame, f"Exercise Completed! {st.session_state.reps} reps", 
                       (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            feedback_container.markdown(f"""
            **Workout Summary:**
            - Repetitions: {st.session_state.reps}
            - Time Elapsed: {duration:.2f} minutes
            - Calories Burned: {calories:.2f} kcal
            """)
            if (st.session_state.current_audio != "completed" and 
                current_time >= st.session_state.audio_end_time):
                audio_base64, duration = audio_cache["completed"]
                play_audio_backend(audio_base64, key=f"completed_{current_time}")
                st.session_state.current_audio = "completed"
                st.session_state.audio_end_time = current_time + duration

        stframe.image(frame, channels="BGR", use_container_width=True)
        if step_idx == 0 or st.session_state.completed:
            debug_container.empty()

        if st.session_state.completed:
            break

    cap.release()
    pose.close()
