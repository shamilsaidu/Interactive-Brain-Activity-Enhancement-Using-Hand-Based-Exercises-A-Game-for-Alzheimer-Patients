import cv2
import csv
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import mediapipe as mp
import numpy as np
from enum import Enum
from app import calc_bounding_rect,calc_landmark_list,pre_process_landmark,pre_process_point_history
import time
import pandas as pd
from dummpy import GestureClassifier
from model import KeyPointClassifier
from tensorflow.keras.models import load_model # type: ignore

class HandState(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    UNKNOWN = "UNKNOWN"

class ExerciseState(Enum):
    WAITING_TO_START = "WAITING_TO_START"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"


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
        self.gesture_classifier = KeyPointClassifier()

        with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                  encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            keypoint_classifier_labels = [
                row[0] for row in keypoint_classifier_labels
            ]

        # Window dimensions
        self.window_width = 1000
        self.window_height = 800

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

        # History tracking
        self.history_length = 16

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
    
    def check_movement_correctness(self):
        """Check if the current hand positions match the required exercise state."""
        if self.left_state == HandState.UNKNOWN or self.right_state == HandState.UNKNOWN:
            return False

        current_state = (self.left_state, self.right_state)
        current_time = time.time()

        # Define the correct states
        state1 = (HandState.CLOSED, HandState.OPEN)
        state2 = (HandState.OPEN, HandState.CLOSED)
        expected_states = [state1, state2]

        # Check for errors if the cooldown has passed
        if current_time - self.last_error_time >= self.error_cooldown:
            # Incorrect sequence
            if current_state != expected_states[self.expected_state_idx]:
                if current_state == expected_states[1 - self.expected_state_idx]:
                    self.error_message = "Wrong sequence! Follow the pattern."
                else:
                    self.error_message = "Incorrect hand positions!"
                self.apply_penalty()
                self.last_error_time = current_time
                self.error_message_time = current_time

            # Moving too slow
            elif current_time - self.last_movement_time > self.movement_timeout:
                self.error_message = "Moving too slow! Keep pace."
                self.apply_penalty()
                self.last_error_time = current_time
                self.error_message_time = current_time

        # Correct state handling
        if current_state == expected_states[self.expected_state_idx]:
            if self.last_valid_state != current_state:
                self.last_valid_state = current_state
                self.expected_state_idx = 1 - self.expected_state_idx
                self.last_movement_time = current_time
                return True

        return False

  
        """Apply a penalty to the user's score."""
        self.score = max(0, self.score - self.penalty_points)

    def determine_hand_state(self, hand_landmarks):
        """Determine if a hand is open or closed based on finger positions"""
        if not hand_landmarks:
            return HandState.UNKNOWN
            
        fingers_open = 0
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip, pip in zip(finger_tips, finger_pips):
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                fingers_open += 1
                
        if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
            fingers_open += 1
            
        return HandState.OPEN if fingers_open >= 3 else HandState.CLOSED

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

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(frame, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(frame, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                handedness = results.multi_handedness[hand_idx].classification[0].label
                hand_state = self.determine_hand_state(hand_landmarks)
                gesture = KeyPointClassifier(pre_processed_landmark_list)
                
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
                cv2.putText(frame, f"Gesture: {gesture}",
                           (cx - 50, cy + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if self.check_movement_correctness():
            self.current_rep += 1
        
        if self.current_rep >= self.required_repetitions:
            self.exercise_state = ExerciseState.COMPLETED
        
        self.draw_exercise_info(frame)
        return frame

    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

         # Convert to a one-dimensional list
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))
    
        def normalize_(n):
            return n / max_value
    
        temp_landmark_list = list(map(normalize_, temp_landmark_list))
    
        return temp_landmark_list

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

        """Check if the current hand positions match the required exercise state"""
        if self.left_state == HandState.UNKNOWN or self.right_state == HandState.UNKNOWN:
            return False
            
        current_state = (self.left_state, self.right_state)
        current_time = time.time()
        
        state1 = (HandState.CLOSED, HandState.OPEN)
        state2 = (HandState.OPEN, HandState.CLOSED)
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