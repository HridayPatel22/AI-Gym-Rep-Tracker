import cv2
import mediapipe as mp
import numpy as np


class PoseEstimator:
    """Class for pose estimation and rep counting using OpenCV and MediaPipe."""

    # Class constants for display
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.8
    FONT_THICKNESS = 2
    ANNOTATION_COLOR = (255, 255, 255)
    BOX_COLOR = (0, 0, 139)
    REPS_Y_POSITION = 75
    STAGE_Y_POSITION = 75

    def __init__(self):
        """Initialize the PoseEstimator with default values and start video capture."""
        self.counter = 0
        self.stage = None
        self.cap = self.initialize_video_capture()
        self.mp_pose = mp.solutions.pose

    @staticmethod
    def initialize_video_capture():
        """Initialize the webcam video capture using the laptop webcam."""
        return cv2.VideoCapture(0)

    def process_frame(self, pose, frame):
        """Process the frame to detect pose landmarks and return the annotated image."""
        # Convert the frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image to detect pose landmarks
        results = pose.process(image)

        # Convert the image back to BGR for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image, results

    @staticmethod
    def draw_landmarks(image, results):
        """Draw pose landmarks on the image."""
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        # Draw the pose landmarks on the image with specified drawing styles
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 105, 65), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 140, 255), thickness=2, circle_radius=2)
        )

    @staticmethod
    def calculate_angle(a, b, c):
        """Calculate the angle between three points: a, b (vertex), and c."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        # Calculate the angle using arctan2 function
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    @staticmethod
    def get_landmarks(results):
        """Extract and return landmarks from pose detection results."""
        return results.pose_landmarks.landmark

    def annotate_angle_image(self, image, angle, position):
        """Annotate the calculated angle on the image at the specified position."""
        cv2.putText(
            image, str(angle),
            position,
            self.FONT, self.FONT_SCALE, self.ANNOTATION_COLOR, self.FONT_THICKNESS, cv2.LINE_AA
        )

    def display_reps_stage(self, image):
        """Display rep counter and current stage of the rep."""
        # Draw a background box and display the rep count and stage on the image
        cv2.rectangle(image, (0, 0), (265, 95), self.BOX_COLOR, -1)
        cv2.putText(image, 'Reps', (15, 30), self.FONT, self.FONT_SCALE, self.ANNOTATION_COLOR, 1, cv2.LINE_AA)
        cv2.putText(image, str(self.counter), (15, self.REPS_Y_POSITION), self.FONT, 1.7, self.ANNOTATION_COLOR, 2, cv2.LINE_AA)
        cv2.putText(image, 'Stage', (120, 30), self.FONT, self.FONT_SCALE, self.ANNOTATION_COLOR, 1, cv2.LINE_AA)
        cv2.putText(image, self.stage, (120, self.STAGE_Y_POSITION), self.FONT, 1.7, self.ANNOTATION_COLOR, 2, cv2.LINE_AA)

    def run(self):
        """Main function to run pose estimation and rep counting."""
        # Initialize the MediaPipe Pose model with detection and confidence thresholds
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Process the frame and get pose landmarks
                image, results = self.process_frame(pose, frame)

                try:
                    landmarks = self.get_landmarks(results)

                    # Extract coordinates for shoulder, elbow, and wrist
                    shoulder = [
                        landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                    ]
                    elbow = [
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y
                    ]
                    wrist = [
                        landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y
                    ]

                    # Calculate the angle at the elbow using the shoulder, elbow, and wrist points
                    angle = self.calculate_angle(shoulder, elbow, wrist)

                    self.annotate_angle_image(image, angle, tuple(np.multiply(elbow, [640, 480]).astype(int)))

                    # Determine the stage of the exercise based on the angle
                    if angle > 160:
                        self.stage = "down"
                    if angle < 30 and self.stage == "down":
                        self.stage = "up"
                        self.counter += 1
                        print(self.counter)

                except AttributeError:
                    pass

                self.display_reps_stage(image)
                self.draw_landmarks(image, results)
                cv2.imshow('Video Feed', image)

                # Exit the loop if the 'q' key is pressed
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # Create an instance of the PoseEstimator class and run the pose estimation
    pose_estimator = PoseEstimator()
    pose_estimator.run()

