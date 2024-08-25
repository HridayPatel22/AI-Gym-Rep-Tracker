import cv2
import mediapipe as mp
import numpy as np

# Constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 2
ANNOTATION_COLOR = (255, 255, 255)
BOX_COLOR = (0, 0, 139)
REPS_Y_POSITION = 75
STAGE_Y_POSITION = 75


def initialize_video_capture():
    """Initialize the webcam video capture using the laptop webcam."""
    return cv2.VideoCapture(0)


def process_frame(pose, frame):
    """Process the frame to detect pose landmarks and return the annotated image."""
    # Convert the image to RGB for processing
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Perform pose detection
    results = pose.process(image)

    # Convert the image back to BGR for display
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, results


def draw_landmarks(image, results):
    """Draw pose landmarks on the image."""
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Draw landmarks with specified colors and thickness
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(255, 105, 65), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0, 140, 255), thickness=2, circle_radius=2)
    )


def calculate_angle(a, b, c):
    """Calculate the angle between the first, middle, and third points."""
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point (vertex of the angle)
    c = np.array(c)  # Third point

    # Calculate the angle using the arctan2 function
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    # Ensure the angle is within the range [0, 180]
    if angle > 180.0:
        angle = 360 - angle

    return angle


def get_landmarks(results):
    """Extract and return landmarks from pose detection results."""
    return results.pose_landmarks.landmark


def annotate_angle_on_image(image, angle, position):
    """Annotate the calculated angle on the image at the specified position."""
    cv2.putText(
        image, str(angle),
        position,
        FONT, FONT_SCALE, ANNOTATION_COLOR, FONT_THICKNESS, cv2.LINE_AA
    )


def display_reps_and_stage(image, counter, stage):
    """Display rep counter and current stage of the rep."""
    # Draw a box for the counter and stage
    cv2.rectangle(image, (0, 0), (265, 95), BOX_COLOR, -1)

    # Display 'Reps' heading and the counter value
    cv2.putText(image, 'Reps', (15, 30), FONT, FONT_SCALE, ANNOTATION_COLOR, 1, cv2.LINE_AA)
    cv2.putText(image, str(counter), (15, REPS_Y_POSITION), FONT, 1.7, ANNOTATION_COLOR, 2, cv2.LINE_AA)

    # Display 'Stage' heading and the stage value
    cv2.putText(image, 'Stage', (120, 30), FONT, FONT_SCALE, ANNOTATION_COLOR, 1, cv2.LINE_AA)
    cv2.putText(image, stage, (120, STAGE_Y_POSITION), FONT, 1.7, ANNOTATION_COLOR, 2, cv2.LINE_AA)


def main():
    """Main function to run pose estimation and rep counting."""
    cap = initialize_video_capture()

    # Initialize variables for the rep counter
    counter = 0
    stage = None

    # Initialize the pose detection system with confidence thresholds
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame and get pose landmarks
            image, results = process_frame(pose, frame)

            try:
                landmarks = get_landmarks(results)

                # Calculate the angle between shoulder, elbow, and wrist
                shoulder = [
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                ]
                elbow = [
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
                ]
                wrist = [
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
                ]

                angle = calculate_angle(shoulder, elbow, wrist)

                # Add the calculated angle to the image
                annotate_angle_on_image(
                    image, angle, tuple(np.multiply(elbow, [640, 480]).astype(int))
                )

                # Update the rep counter based on the angle
                if angle > 160:
                    stage = "down"
                if angle < 30 and stage == "down":
                    stage = "up"
                    counter += 1
                    print(counter)

            except AttributeError:
                pass

            # Display the rep counter and stage on the image
            display_reps_and_stage(image, counter, stage)

            # Draw pose landmarks on the image
            draw_landmarks(image, results)

            # Display the image
            cv2.imshow('Video Feed', image)

            # Exit the loop if 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
