import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
import pyautogui

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
class MouseController:
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()
        # Smoothing factor: higher = smoother but more lag
        self.smoothening = 5.0 
        self.prev_x, self.prev_y = self.screen_w / 2, self.screen_h / 2
        self.curr_x, self.curr_y = self.screen_w / 2, self.screen_h / 2
        
        self.click_cooldown = 0
        self.mode = "MOVE"
        self.mode_flicker_count = 0
        self.stable_mode = "MOVE"
        
        self.is_drawing = False
        self.movement_pause = 0
        self.dead_zone = 8
        self.is_pinching_click = False
        self.pinch_frames = 0
        self.palm_frames = 0
        self.fist_cooldown = 0
        self.is_pinched = False
        
    def determine_mode(self, fingers_up, is_pinched):
        up_count = sum(fingers_up)
        
        # Priority System
        if up_count == 4:
            new_mode = "PALM"
        elif up_count == 0 and not is_pinched: # FIST only if not actively pinching
            new_mode = "FIST"
        else:
            new_mode = "DRAW"
            
        if new_mode != self.mode:
            self.mode = new_mode
            self.mode_flicker_count = 1
        else:
            self.mode_flicker_count += 1
            
        # Add frame buffering for mode switching (debounce flicker)
        if self.mode_flicker_count > 7:
            self.stable_mode = self.mode
            
        # Exception: if we are actively pinching, ensure we stay in DRAW mode
        if is_pinched and self.stable_mode != "DRAW":
            self.stable_mode = "DRAW"

    def update(self, index_pt, thumb_pt, cam_w, cam_h, fingers_up):
        # Evaluate pinch state globally for the frame FIRST
        dist = 0
        is_pinched = False
        if index_pt and thumb_pt:
            dist = math.hypot(thumb_pt[0] - index_pt[0], thumb_pt[1] - index_pt[1])
            if dist < 40:
                self.pinch_frames += 1
            else:
                self.pinch_frames = 0
            
            is_pinched = self.pinch_frames > 2 # Require multiple frames of pinch to confirm
            
        self.is_pinched = is_pinched
        
        # Determine mode dynamically with pinch priority
        self.determine_mode(fingers_up, is_pinched)
        
        # 1. Map and Move (allowed in all modes so cursor doesn't freeze)
        if index_pt:
            # Expand camera bounds mapping to make movement feel less sensitive
            target_x = np.interp(index_pt[0], [cam_w * 0.05, cam_w * 0.95], [0, self.screen_w])
            target_y = np.interp(index_pt[1], [cam_h * 0.05, cam_h * 0.95], [0, self.screen_h])
            
            # Dead zone: ignore micro-movements to reduce shivering
            dist_to_target = math.hypot(target_x - self.prev_x, target_y - self.prev_y)
            if dist_to_target < self.dead_zone:
                target_x, target_y = self.prev_x, self.prev_y

            # Fix drawing release jitter delay
            if self.movement_pause > 0:
                self.movement_pause -= 1
                target_x, target_y = self.prev_x, self.prev_y
                
            self.curr_x = self.prev_x + (target_x - self.prev_x) / self.smoothening
            self.curr_y = self.prev_y + (target_y - self.prev_y) / self.smoothening
            
            clamped_x = self.curr_x
            clamped_y = self.curr_y
            
            # Prevent cursor from entering the Top-Right UI region (400x300 zone + 10px buffer)
            if clamped_x > self.screen_w - 410 and clamped_y < 310:
                # Push back out of the zone depending on which way was closer
                overlap_x = clamped_x - (self.screen_w - 410)
                overlap_y = 310 - clamped_y
                if overlap_x < overlap_y:
                    clamped_x = self.screen_w - 410
                else:
                    clamped_y = 310
                    
            # Safe Zone Limiters: Block cursor from hitting taskbar or sharp window corners
            clamped_x = max(20, min(self.screen_w - 20, clamped_x))
            clamped_y = max(20, min(self.screen_h - 60, clamped_y)) # 60px bottom protection margin
            
            pyautogui.moveTo(int(clamped_x), int(clamped_y))
            self.prev_x, self.prev_y = clamped_x, clamped_y
            
        if self.fist_cooldown > 0:
            self.fist_cooldown -= 1
            
        # 2. Control Mode Routing
        if self.stable_mode == "DRAW":
            self.palm_frames = 0 # reset hold timer
            if is_pinched:
                if not self.is_drawing:
                    pyautogui.mouseDown()
                    self.is_drawing = True
            else:
                if self.is_drawing:
                    pyautogui.mouseUp()
                    self.is_drawing = False
                    self.movement_pause = 10 
                    
        elif self.stable_mode == "FIST":
            self.palm_frames = 0
            if self.is_drawing:
                pyautogui.mouseUp()
                self.is_drawing = False
                
            # Keep object delete fast and responsive
            if self.fist_cooldown == 0:
                pyautogui.press('delete')
                self.fist_cooldown = 40 # Prevent spamming delete
                
        elif self.stable_mode == "PALM":
            if self.is_drawing:
                pyautogui.mouseUp()
                self.is_drawing = False
                
            # Add delay (frame hold) for open palm to prevent accidental full clear
            self.palm_frames += 1
            if self.palm_frames == 40: # Approx 1.5 seconds hold
                pyautogui.hotkey('ctrl', 'a')
                pyautogui.press('delete')

class HandTracker:
    def __init__(self, max_hands=1):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def get_hand_info(self, frame):
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        h, w, c = frame.shape
        fingertips = {}
        fingers_up = [False, False, False, False]
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Landmark IDs for all 5 fingertips
                finger_ids = [4, 8, 12, 16, 20]
                for fid in finger_ids:
                    lm = hand_landmarks.landmark[fid]
                    fingertips[fid] = (int(lm.x * w), int(lm.y * h))
                
                # Check up states: tip y < pip y
                lmps = hand_landmarks.landmark
                fingers_up[0] = lmps[8].y < lmps[6].y
                fingers_up[1] = lmps[12].y < lmps[10].y
                fingers_up[2] = lmps[16].y < lmps[14].y
                fingers_up[3] = lmps[20].y < lmps[18].y

                break # Ensure we only track one hand
                
        return fingertips, fingers_up

class EffectsRenderer:
    def __init__(self):
        self.finger_ids = [4, 8, 12, 16, 20]
        # Trail configuration (limits particle count) per finger
        self.trail_points = {fid: deque(maxlen=20) for fid in self.finger_ids} 
        
        # Unique colors per finger (BGR format)
        self.finger_colors = {
            4: (0, 150, 255),    # Thumb: Orange
            8: (255, 255, 0),    # Index: Cyan
            12: (0, 255, 0),     # Middle: Green
            16: (255, 0, 255),   # Ring: Magenta
            20: (0, 0, 255)      # Pinky: Red
        }
        
        # Standard time offset for continuous animation
        self.time_offset = 0.0
        
        # Pre-calculate points for the 3D sphere (Fibonacci sphere distribution)
        self.num_sphere_pts = 100
        self.sphere_points = self._generate_sphere_points(self.num_sphere_pts)

    def _generate_sphere_points(self, samples):
        # Uniformly distribute points on the surface of a sphere
        points = []
        phi = math.pi * (3. - math.sqrt(5.))  # Golden angle
        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # Y spans from 1 to -1
            radius = math.sqrt(1 - y * y) 
            theta = phi * i  
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            points.append((x, y, z))
        return points

    def update(self, fingertips):
        # Add a new point to the trail, deque automatically removes the oldest
        for fid in self.finger_ids:
            pt = fingertips.get(fid)
            if pt:
                self.trail_points[fid].appendleft(pt)
            else:
                # Gradually shrink trail if finger is not detected
                if len(self.trail_points[fid]) > 0:
                    self.trail_points[fid].pop() 
                
        self.time_offset += 0.1 # Advance time for animations

    def draw_trail(self, frame):
        # Draw fading particle trails for each finger
        for fid, trail in self.trail_points.items():
            base_color = self.finger_colors[fid]
            for i, pt in enumerate(trail):
                # Calculate size based on age
                age_factor = 1.0 - (i / len(trail))
                radius = int(6 * age_factor)
                
                # Fetch color fading for this specific finger
                b = int(base_color[0] * age_factor)
                g = int(base_color[1] * age_factor)
                r = int(base_color[2] * age_factor)
                
                cv2.circle(frame, pt, radius, (b, g, r), -1)

    def draw_spiral(self, frame, center, fid):
        # Draw a continuous smooth spiral line
        points = []
        num_turns = 3
        max_radius = 50
        segments = 40
        
        base_color = self.finger_colors[fid]
        
        # Generate spiral points
        for i in range(segments):
            t = i / segments # Normalized distance
            # Angle includes time_offset for rotation
            angle = t * math.pi * 2 * num_turns + self.time_offset * (1.5 if fid % 2 == 0 else -1.5)
            radius = t * max_radius
            
            x = int(center[0] + math.cos(angle) * radius)
            y = int(center[1] + math.sin(angle) * radius)
            points.append((x, y))
        
        # Connect points to make it continuous and smooth
        if len(points) > 1:
            for i in range(len(points) - 1):
                fade = 1.0 - (i / segments)
                color = (int(base_color[0]*fade), int(base_color[1]*fade), int(base_color[2]*fade))
                cv2.line(frame, points[i], points[i+1], color, 2)

    def draw_sphere(self, frame, center):
        # Rotation matrices for X and Y axes
        cos_t = math.cos(self.time_offset * 0.5)
        sin_t = math.sin(self.time_offset * 0.5)
        
        sphere_radius = 60
        focal_length = 200 # Controls perspective simulation
        
        for pt in self.sphere_points:
            x, y, z = pt
            
            # 3D Rotation Math
            # Rotate around Y axis
            x_rot = x * cos_t - z * sin_t
            z_rot = x * sin_t + z * cos_t
            
            # Rotate around X axis
            y_rot = y * cos_t - z_rot * sin_t
            z_rot_final = y * sin_t + z_rot * cos_t
            
            # Simulate depth with perspective projection
            z_translated = z_rot_final * sphere_radius + focal_length
            
            if z_translated < 1: z_translated = 1
            scale = focal_length / z_translated
            
            screen_x = int(center[0] + x_rot * sphere_radius * scale)
            screen_y = int(center[1] + y_rot * sphere_radius * scale)
            
            # Depth cue: Z value (front to back) mapped to radius and color intensity
            depth_factor = (z_rot_final + 1) / 2 # Normalized 0.0 to 1.0
            pt_radius = int(1 + 4 * depth_factor * scale)
            
            color_intensity = int(80 + 175 * depth_factor)
            color = (0, color_intensity, color_intensity) # Yellowish color
            
            cv2.circle(frame, (screen_x, screen_y), pt_radius, color, -1)

def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    renderer = EffectsRenderer()
    mouse_ctrl = MouseController()

    window_name = "Hand Tracking Magic"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 400, 300)
    # Move to top-right corner safely
    cv2.moveWindow(window_name, mouse_ctrl.screen_w - 400, 0)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Mirror image for intuitive interaction
        frame = cv2.flip(frame, 1) 
        h, w, _ = frame.shape
        
        # Extract finger tips and get state array
        fingertips, fingers_up = tracker.get_hand_info(frame)
        renderer.update(fingertips)
        
        # Mouse Control layer
        index_pt = fingertips.get(8)
        thumb_pt = fingertips.get(4)
        if index_pt:
            mouse_ctrl.update(index_pt, thumb_pt, w, h, fingers_up)
            
            # Visual feedback for clicking (turns index finger green when pinched)
            if thumb_pt and math.hypot(thumb_pt[0] - index_pt[0], thumb_pt[1] - index_pt[1]) < 40:
                cv2.circle(frame, index_pt, 20, (0, 255, 0), cv2.FILLED)

        if fingertips:
            # Render visual effects
            renderer.draw_trail(frame)
            
            for fid, pt in fingertips.items():
                renderer.draw_spiral(frame, pt, fid)
                
                # Highlight the fingertip itself
                cv2.circle(frame, pt, 6, (255, 255, 255), -1)
                cv2.circle(frame, pt, 8, (0, 0, 0), 2)
                
            # Keep sphere just on index finger to avoid UI clutter
            if 8 in fingertips:
                renderer.draw_sphere(frame, fingertips[8])

        # Draw Mode Text
        cv2.putText(frame, f"Mode: {mouse_ctrl.stable_mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        pinch_color = (0, 255, 0) if mouse_ctrl.is_pinched else (0, 0, 255)
        cv2.putText(frame, f"Pinched: {mouse_ctrl.is_pinched}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, pinch_color, 2)

        # Show final frame
        cv2.imshow(window_name, frame)
        
        # Break loop with 'ESC'
        if cv2.waitKey(1) & 0xFF == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
