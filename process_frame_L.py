# process_frame_L.py

import time
import cv2
import numpy as np
from utils_L import draw_text, draw_dotted_line, find_angle, get_landmark_features, get_mediapipe_pose

# Rest of the code remains unchanged


class ProcessFrame_L:
    def __init__(self, thresholds, flip_frame=False):
        self.flip_frame = flip_frame
        self.thresholds = thresholds
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.linetype = cv2.LINE_AA
        self.COLORS = {
            'blue': (0, 127, 255),
            'red': (255, 50, 50),
            'green': (0, 255, 127),
            'light_green': (100, 233, 127),
            'yellow': (255, 255, 0),
            'magenta': (255, 0, 255),
            'white': (255, 255, 255),
            'cyan': (0, 255, 255),
            'light_blue': (102, 204, 255)
        }
        self.dict_features = {}
        self.left_features = {
            'shoulder': 11,
            'elbow': 13,
            'wrist': 15,
            'hip': 23,
            'knee': 25,
            'ankle': 27,
            'foot': 31
        }
        self.right_features = {
            'shoulder': 12,
            'elbow': 14,
            'wrist': 16,
            'hip': 24,
            'knee': 26,
            'ankle': 28,
            'foot': 32
        }
        self.dict_features['left'] = self.left_features
        self.dict_features['right'] = self.right_features
        self.dict_features['nose'] = 0

        self.state_tracker = {
            'state_seq': [],
            'start_inactive_time': time.perf_counter(),
            'INACTIVE_TIME': 0.0,
            'DISPLAY_TEXT': np.full((3,), False),
            'COUNT_FRAMES': np.zeros((3,), dtype=np.int64),
            'LUNGE_COUNT': 0,
            'IMPROPER_LUNGE': 0,
            'prev_state': None,
            'curr_state': None,
        }

        self.FEEDBACK_ID_MAP = {
            0: ('BEND TOO FORWARD', 215, (0, 153, 255)),
            1: ('KNEE OVER TOE', 170, (255, 80, 80)),
            2: ('NOT DEEP ENOUGH', 125, (255, 80, 80))
        }

    def _get_state(self, knee_angle, hip_angle):
        if self.thresholds['knee_angle_min'] <= knee_angle <= self.thresholds['knee_angle_max']:
            if self.thresholds['hip_angle_min'] <= hip_angle <= self.thresholds['hip_angle_max']:
                return 'normal'
            else:
                return 'hip_angle_error'
        else:
            return 'knee_angle_error'

    def _update_state_sequence(self, state):
        if state not in self.state_tracker['state_seq']:
            self.state_tracker['state_seq'].append(state)

    def _show_feedback(self, frame, c_frame, dict_maps):
        for idx in np.where(c_frame)[0]:
            draw_text(
                frame,
                dict_maps[idx][0],
                pos=(30, dict_maps[idx][1]),
                text_color=(255, 255, 230),
                font_scale=0.6,
                font_thickness=2,
                text_color_bg=dict_maps[idx][2]
            )
        return frame

    def process(self, frame: np.array, pose):
        play_sound = None
        frame_height, frame_width, _ = frame.shape

        keypoints = pose.process(frame)

        if keypoints.pose_landmarks:
            ps_lm = keypoints.pose_landmarks

            left_shldr_coord, left_hip_coord, left_knee_coord, left_ankle_coord = \
                get_landmark_features(ps_lm.landmark, self.dict_features, 'left', frame_width, frame_height, features=['shoulder', 'hip', 'knee', 'ankle'])
            right_shldr_coord, right_hip_coord, right_knee_coord, right_ankle_coord = \
                get_landmark_features(ps_lm.landmark, self.dict_features, 'right', frame_width, frame_height, features=['shoulder', 'hip', 'knee', 'ankle'])

            shldr_coord, hip_coord, knee_coord, ankle_coord = (left_shldr_coord, left_hip_coord, left_knee_coord, left_ankle_coord) \
                if left_hip_coord[1] < right_hip_coord[1] else (right_shldr_coord, right_hip_coord, right_knee_coord, right_ankle_coord)

            hip_angle = find_angle(shldr_coord, hip_coord, knee_coord)
            knee_angle = find_angle(hip_coord, knee_coord, ankle_coord)

            current_state = self._get_state(knee_angle, hip_angle)
            self.state_tracker['curr_state'] = current_state
            self._update_state_sequence(current_state)

            frame = self._show_feedback(frame, self.state_tracker['COUNT_FRAMES'], self.FEEDBACK_ID_MAP)

            if current_state == 'normal':
                if len(self.state_tracker['state_seq']) == 1:
                    self.state_tracker['LUNGE_COUNT'] += 1
                    play_sound = str(self.state_tracker['LUNGE_COUNT'])
                self.state_tracker['state_seq'] = []
            else:
                if current_state == 'knee_angle_error':
                    self.state_tracker['DISPLAY_TEXT'][1] = True
                    self.state_tracker['IMPROPER_LUNGE'] += 1
                elif current_state == 'hip_angle_error':
                    self.state_tracker['DISPLAY_TEXT'][0] = True
                    self.state_tracker['IMPROPER_LUNGE'] += 1

            self.state_tracker['COUNT_FRAMES'][self.state_tracker['DISPLAY_TEXT']] += 1
            self.state_tracker['DISPLAY_TEXT'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = False
            self.state_tracker['COUNT_FRAMES'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = 0

            cv2.putText(frame, f'Correct Lunges: {self.state_tracker["LUNGE_COUNT"]}', (10, 30), self.font, 0.8, self.COLORS['green'], 2, lineType=self.linetype)
            cv2.putText(frame, f'Incorrect Lunges: {self.state_tracker["IMPROPER_LUNGE"]}', (10, 60), self.font, 0.8, self.COLORS['red'], 2, lineType=self.linetype)

        else:
            if self.flip_frame:
                frame = cv2.flip(frame, 1)

            end_time = time.perf_counter()
            self.state_tracker['INACTIVE_TIME'] += end_time - self.state_tracker['start_inactive_time']
            if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']:
                self.state_tracker['LUNGE_COUNT'] = 0
                self.state_tracker['IMPROPER_LUNGE'] = 0
                play_sound = 'reset_counters'
            self.state_tracker['start_inactive_time'] = end_time

        return frame, play_sound
