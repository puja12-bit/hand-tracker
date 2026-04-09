import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque

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

    def get_index_fingertip(self, frame):
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        h, w, c = frame.shape
        fingertip = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Landmark 8 is the tip of the index finger
                lm = hand_landmarks.landmark[8]
                fingertip = (int(lm.x * w), int(lm.y * h))
                break # Ensure we only track one hand
                
        return fingertip

class EffectsRenderer:
    def __init__(self):
        # Trail configuration (limits particle count to 30)
        self.trail_points = deque(maxlen=30) 
        
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

    def update(self, fingertip):
        # Add a new point to the trail, deque automatically removes the oldest
        if fingertip:
            self.trail_points.appendleft(fingertip)
        else:
            # Gradually shrink trail if finger is not detected
            if len(self.trail_points) > 0:
                self.trail_points.pop() 
                
        self.time_offset += 0.1 # Advance time for animations

    def draw_trail(self, frame):
        # Draw a fading particle trail
        for i, pt in enumerate(self.trail_points):
            # Calculate size based on age
            age_factor = 1.0 - (i / len(self.trail_points))
            radius = int(6 * age_factor)
            
            # Color variation: Green to Blue based on age
            b = 255
            g = int(255 * age_factor)
            r = 0
            
            cv2.circle(frame, pt, radius, (b, g, r), -1)

    def draw_spiral(self, frame, center):
        # Draw a continuous smooth spiral line
        points = []
        num_turns = 3
        max_radius = 80
        segments = 60
        
        # Generate spiral points
        for i in range(segments):
            t = i / segments # Normalized distance
            # Angle includes time_offset for rotation
            angle = t * math.pi * 2 * num_turns + self.time_offset * 1.5
            radius = t * max_radius
            
            x = int(center[0] + math.cos(angle) * radius)
            y = int(center[1] + math.sin(angle) * radius)
            points.append((x, y))
        
        # Connect points to make it continuous and smooth
        if len(points) > 1:
            for i in range(len(points) - 1):
                # Color variation along the spiral path (Cyan to Magenta)
                color = (255, int(255 * (1 - i/segments)), 255) 
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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Mirror image for intuitive interaction
        frame = cv2.flip(frame, 1) 
        
        # Extract finger tip and update rendering states
        fingertip = tracker.get_index_fingertip(frame)
        renderer.update(fingertip)
        
        if fingertip:
            # Render visual effects
            renderer.draw_trail(frame)
            renderer.draw_spiral(frame, fingertip)
            renderer.draw_sphere(frame, fingertip)
            
            # Highlight the fingertip itself
            cv2.circle(frame, fingertip, 6, (255, 255, 255), -1)
            cv2.circle(frame, fingertip, 8, (0, 0, 0), 2)

        # Show final frame
        cv2.imshow("Hand Tracking Magic", frame)
        
        # Break loop with 'ESC'
        if cv2.waitKey(1) & 0xFF == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
