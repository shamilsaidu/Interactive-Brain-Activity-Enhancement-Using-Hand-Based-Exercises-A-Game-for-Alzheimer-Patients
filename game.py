import cv2
import mediapipe as mp
import numpy as np
from enum import Enum
import time
import pandas as pd
from tensorflow.keras.models import load_model # type: ignore

class HandState(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    POINTER = "POINTER"
    OK ="OK"
    PEACE ="PEACE"
    ROCK ="ROCK"
    GUN = "GUN"
    UNKNOWN = "UNKNOWN"

class ExerciseState(Enum):
    WAITING_TO_START = "WAITING_TO_START"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"

class GestureClassifier:
    def __init__(self, model_path='model\keypoint_classifier\keypoint_classifier.keras', labels_path='model\keypoint_classifier\keypoint_classifier_label.csv'):
        self.model = load_model(model_path)
        self.labels = pd.read_csv(labels_path, header=None)[0].tolist()
    
    def pre_process_landmark(self, landmark_list):
        """Convert landmark coordinates to relative coordinates and normalize"""
        temp_landmark_list = []
        base_x, base_y = 0, 0

        for index, landmark_point in enumerate(landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list.append([landmark_point[0] - base_x, landmark_point[1] - base_y])

        max_value = max(list(map(abs, sum(temp_landmark_list, []))))
        def normalize_(n):
            return n / max_value if max_value != 0 else 0

        normalized_landmark_list = list(map(lambda p: [normalize_(p[0]), normalize_(p[1])], temp_landmark_list))
        return np.array(normalized_landmark_list).flatten()

    def predict(self, landmarks):
        """Predict gesture from landmarks"""
        landmark_list = [[lm.x, lm.y] for lm in landmarks.landmark]
        preprocessed = self.pre_process_landmark(landmark_list)
        prediction = self.model.predict(np.array([preprocessed]), verbose=0)
        return self.labels[np.argmax(prediction)], np.max(prediction)

class ExerciseTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.gesture_classifier = GestureClassifier()
        
        # Window dimensions
        self.window_width = 1000  # Increased from default
        self.window_height = 800  # Increased from default
        
        # Exercise state
        self.exercise_state = ExerciseState.WAITING_TO_START
        
        # Exercise parameters
        self.required_repetitions = 10
        self.current_rep = 0
        self.score = 100
        self.penalty_points = 4
        self.passing_score = 75
        
        # State tracking
        self.left_state = HandState.UNKNOWN
        self.right_state = HandState.UNKNOWN
        self.last_valid_state = None
        
        # Gesture tracking
        self.left_gesture = ""
        self.right_gesture = ""
        
        # Error detection parameters
        self.expected_state_idx = 0
        self.error_cooldown = 1.0
        self.last_error_time = time.time()
        self.last_movement_time = time.time()
        self.movement_timeout = 3.0
        self.error_message = ""
        self.error_message_duration = 2.0
        self.error_message_time = 0

    def draw_guidance_screen(self, frame):
        """Draw the initial guidance screen"""
        # Resize frame to match desired dimensions
        frame = cv2.resize(frame, (self.window_width, self.window_height))
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.window_width, self.window_height), (0, 0, 0), -1)
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Title
        title_font_size = 1.5
        title_y = 80
        cv2.putText(frame, "Hand Movement Exercise Guide",
                   (self.window_width//2 - 300, title_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   title_font_size, 
                   (255, 255, 255), 
                   2)

        # Instructions
        instructions = [
            "Exercise Pattern:",
            "1. Start with LEFT hand CLOSED, RIGHT hand OPEN",
            "2. Then alternate to LEFT hand OPEN, RIGHT hand CLOSED",
            "3. Repeat this pattern for 10 repetitions",
            "",
            "Important Rules:",
            "- Maintain a steady pace (within 3 seconds per movement)",
            "- Follow the correct sequence of movements",
            "- Keep your hands clearly visible to the camera",
            "",
            "Scoring System:",
            "- Starting score: 100 points",
            "- Penalty for incorrect movements: -4 points",
            "- Penalty for slow movements: -4 points",
            "- Minimum passing score: 75 points",
            "",
            "Controls:",
            "Press 'S' to start the exercise",
            "Press 'Q' to quit at any time"
        ]

        # Calculate text positioning
        font_size = 0.8
        line_height = 35
        start_y = 150
        left_margin = 100

        # Draw instructions with improved spacing
        for i, instruction in enumerate(instructions):
            y_position = start_y + (i * line_height)
            
            # Add section highlighting
            if instruction.endswith(":"):
                # Section headers in yellow
                color = (0, 255, 255)
                font_size_header = 1.0
                cv2.putText(frame, instruction,
                           (left_margin, y_position), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           font_size_header, 
                           color, 
                           2)
            else:
                # Regular text in white
                color = (255, 255, 255)
                cv2.putText(frame, instruction,
                           (left_margin + 20, y_position), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           font_size, 
                           color, 
                           2)

        return frame

    def determine_hand_state(self, hand_landmarks):

        if not hand_landmarks:
            return HandState.UNKNOWN
    
        # Use the gesture classifier to predict the gesture and confidence
        gesture, confidence = self.gesture_classifier.predict(hand_landmarks)
        
        # Map the predicted gesture to a HandState
        gesture_to_hand_state = {
            "Open": HandState.OPEN,
            "Close": HandState.CLOSED,
            "Pointer": HandState.POINTER,
            "OK": HandState.OK,
            "peace": HandState.PEACE,
            "rock": HandState.ROCK,
            "gun": HandState.GUN,
        }
        
        # Return the corresponding hand state or UNKNOWN if the gesture is not recognized
        return gesture_to_hand_state.get(gesture, HandState.UNKNOWN)


    def process_frame(self, frame):
        """Process a single frame and update exercise state"""
        # Resize frame to match desired dimensions
        frame = cv2.resize(frame, (self.window_width, self.window_height))
        
        if self.exercise_state == ExerciseState.WAITING_TO_START:
            return self.draw_guidance_screen(frame)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        # Reset states for this frame
        self.left_state = HandState.UNKNOWN
        self.right_state = HandState.UNKNOWN
        self.left_gesture = ""
        self.right_gesture = ""
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                handedness = results.multi_handedness[hand_idx].classification[0].label
                hand_state = self.determine_hand_state(hand_landmarks)
                gesture, confidence = self.gesture_classifier.predict(hand_landmarks)
                
                if handedness == 'Left':
                    self.left_state = hand_state
                    self.left_gesture = gesture
                else:
                    self.right_state = hand_state
                    self.right_gesture = gesture
                
                h, w, _ = frame.shape
                cx = int(hand_landmarks.landmark[0].x * w)
                cy = int(hand_landmarks.landmark[0].y * h)
                cv2.putText(frame, f"{handedness}: {hand_state.value}",
                           (cx - 50, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Gesture: {gesture} ({confidence:.2f})",
                           (cx - 50, cy + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if self.check_movement_correctness():
            self.current_rep += 1
        
        if self.current_rep >= self.required_repetitions:
            self.exercise_state = ExerciseState.COMPLETED
        
        self.draw_exercise_info(frame)
        return frame

    def draw_exercise_info(self, frame):
        """Draw exercise information on the frame"""
        # Adjust text positioning for larger window
        cv2.putText(frame, f"Repetitions: {self.current_rep}/{self.required_repetitions}",
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Score: {self.score}", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Left Gesture: {self.left_gesture}", (50, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Right Gesture: {self.right_gesture}", (50, 190),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if time.time() - self.error_message_time < self.error_message_duration:
            cv2.putText(frame, self.error_message, (50, 230),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if self.exercise_state == ExerciseState.COMPLETED:
            result = "PASS" if self.score >= self.passing_score else "FAIL"
            cv2.putText(frame, f"Exercise Complete! {result}",
                       (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    def check_movement_correctness(self):
        """Check if the current hand positions match the required exercise state"""
        if self.left_state == HandState.UNKNOWN or self.right_state == HandState.UNKNOWN:
            return False
            
        current_state = (self.left_state, self.right_state)
        current_time = time.time()
        
        state1 = (HandState.POINTER, HandState.PEACE)
        state2 = (HandState.PEACE, HandState.POINTER)
        expected_states = [state1, state2]
        
        if current_time - self.last_error_time >= self.error_cooldown:
            if current_state != expected_states[self.expected_state_idx]:
                if current_state == expected_states[1 - self.expected_state_idx]:
                    self.error_message = "Wrong sequence! Follow the pattern."
                else:
                    self.error_message = "Incorrect hand positions!"
                self.apply_penalty()
                self.last_error_time = current_time
                self.error_message_time = current_time
            
            elif current_time - self.last_movement_time > self.movement_timeout:
                self.error_message = "Moving too slow! Keep pace."
                self.apply_penalty()
                self.last_error_time = current_time
                self.error_message_time = current_time
        
        if current_state == expected_states[self.expected_state_idx]:
            if self.last_valid_state != current_state:
                self.last_valid_state = current_state
                self.expected_state_idx = 1 - self.expected_state_idx
                self.last_movement_time = current_time
                return True
                
        return False

    def apply_penalty(self):
        """Apply score penalty and ensure score doesn't go below 0"""
        self.score = max(0, self.score - self.penalty_points)

def main():
    cap = cv2.VideoCapture(0)
    tracker = ExerciseTracker()
    
    # Set capture resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, tracker.window_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, tracker.window_height)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame = cv2.flip(frame, 1)
        frame = tracker.process_frame(frame)
        
        # Create named window with specific size
        cv2.namedWindow('Hand Movement Exercise', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Hand Movement Exercise', tracker.window_width, tracker.window_height)
        cv2.imshow('Hand Movement Exercise', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and tracker.exercise_state == ExerciseState.WAITING_TO_START:
            tracker.exercise_state = ExerciseState.RUNNING
        elif tracker.exercise_state == ExerciseState.COMPLETED:
            time.sleep(3)  # Show the final result for 3 seconds
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()