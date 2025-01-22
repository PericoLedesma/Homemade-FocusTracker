import cv2
import mediapipe as mp
import time
import face_recognition

from utils import *

# -----------------------
# GLOBAL SETTINGS
# -----------------------
# Thresholds for YOLO detection
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# Minimum distance (in pixels) to consider a "drink" (hand near cup/bottle)
DRINK_DISTANCE_THRESHOLD = 50

# Focus/Pause thresholds
FOCUS_THRESHOLD = 5  # Consecutive detections to consider "Focus"
PAUSE_THRESHOLD = 5  # Consecutive detections to consider "Pause"

# ------------------------------------------------------------------
# LOAD YOLO MODEL AND COCO CLASSES
# ------------------------------------------------------------------
def load_yolo_models():
    """
    Loads the YOLO network (config + weights) and its classes.
    Returns (net, output_layers, classes_list).
    """

    YOLO_DIR = "yolo"
    YOLO_CONFIG = os.path.join(YOLO_DIR, "yolov3.cfg")
    YOLO_WEIGHTS = os.path.join(YOLO_DIR, "yolov3.weights")
    YOLO_CLASSES = os.path.join(YOLO_DIR, "coco.names")

    # Verify that needed files exist
    verify_files_exist([YOLO_CONFIG, YOLO_WEIGHTS, YOLO_CLASSES])

    # Load COCO classes
    with open(YOLO_CLASSES, "r") as f:
        classes_list = [line.strip() for line in f.readlines()]

    # Load the YOLO network
    net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return net, output_layers, classes_list


# ------------------------------------------------------------------
#  INITIALIZE MODELS FOR MEDIA PIPE (POSE, HANDS, FACEMESH)
# ------------------------------------------------------------------
def init_pose_model():
    """
    Initializes the MediaPipe Pose model.
    """
    mp_pose = mp.solutions.pose
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

def init_hand_model():
    """
    Initializes the MediaPipe Hands detection model.
    """
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

def init_face_mesh_model():
    """
    Initializes the MediaPipe Face Mesh model (optional if needed).
    """
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )


# ------------------------------------------------------------------
#  DETECTION FUNCTIONS
# ------------------------------------------------------------------
def detect_pose(img, pose_model, mp_draw_pose) -> bool:
    """
    Detects a body pose in the image using MediaPipe Pose.
    Returns True if pose_landmarks are found, indicating a person's presence.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose_model.process(img_rgb)
    if result.pose_landmarks:
        mp_draw_pose.draw_landmarks(img, result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        return True
    return False


def detect_objects(img, net, output_layers, classes_list):
    """
    Detects objects in the image using YOLO.
    Returns:
      - the frame with bounding boxes drawn
      - a list of tuples (label, (x, y, w, h)) for the objects of interest:
        "bottle", "cup", "cupboard", "cell phone".
    """
    h, w, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416),
                                 (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []

    # Filter detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE_THRESHOLD:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                bw = int(detection[2] * w)
                bh = int(detection[3] * h)
                x = int(center_x - bw / 2)
                y = int(center_y - bh / 2)

                boxes.append([x, y, bw, bh])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-Max Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    detected_objects = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            label = classes_list[class_ids[i]]
            if label in ["bottle", "cup", "cupboard", "cell phone"]:
                x, y, bw, bh = boxes[i]
                detected_objects.append((label, (x, y, bw, bh)))

                # Draw bounding box and label
                color = (255, 0, 0) if label == "cell phone" else (0, 255, 0)
                cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 2)
                cv2.putText(img, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img, detected_objects


def detect_hand(img, hand_model, mp_draw) -> tuple:
    """
    Detects a hand in the image using MediaPipe Hands.
    Returns:
      - the image with the hand drawn
      - a boolean 'hand_present'
      - (index_x, index_y): coordinates of the index fingertip, or (None, None) if not found.
    """
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hand_model.process(img_rgb)

    hand_present = False
    index_coords = (None, None)

    if result.multi_hand_landmarks:
        for hand_lm in result.multi_hand_landmarks:
            hand_present = True
            # Index fingertip is landmark #8
            index_tip = hand_lm.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            index_coords = (index_x, index_y)

            # Draw hand landmarks
            mp_draw.draw_landmarks(img, hand_lm, mp.solutions.hands.HAND_CONNECTIONS)
            cv2.circle(img, index_coords, 5, (255, 0, 0), -1)
            break  # Process the first detected hand only
    return img, hand_present, index_coords


# ------------------------------------------------------------------
#  FACIAL CALIBRATION
# ------------------------------------------------------------------
def load_pedro_encodings(duration=5):
    """
    Activates the camera for 'duration' seconds to capture Pedro's face encodings.
    Returns the list of encodings collected.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access camera.")
        exit(1)

    encodings = []
    start_time = time.time()
    print("Initial calibration: keep your face in front of the camera...")

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame during calibration.")
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb_frame)
        face_encs = face_recognition.face_encodings(rgb_frame, face_locs)

        if face_encs and face_locs:
            encodings += face_encs
            # Draw rectangle around the first face found
            top, right, bottom, left = face_locs[0]
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 2)
            cv2.putText(frame, "Calibrating...", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        cv2.imshow("Initial Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Calibration interrupted by user.")
            break

    cap.release()
    cv2.destroyWindow("Initial Calibration")

    if not encodings:
        print("Error: No face was detected during calibration.")
        exit(1)

    print(f"Calibration done. Captured {len(encodings)} face encodings.")
    return encodings

# ------------------------------------------------------------------
#  MAIN FUNCTION
# ------------------------------------------------------------------
def main():
    # 1) Initial Facial Calibration
    pedro_encs = load_pedro_encodings(duration=5)
    # Take the average of Pedro's encodings
    pedro_encoding = np.mean(pedro_encs, axis=0)

    # 2) Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access camera.")
        exit(1)

    # Optional: set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 3) Load YOLO model and MediaPipe models
    net, output_layers, classes_list = load_yolo_models()
    pose_model = init_pose_model()
    hand_model = init_hand_model()
    # face_mesh_model = init_face_mesh_model()  # If needed

    mp_draw_pose = mp.solutions.drawing_utils
    mp_draw_hands = mp.solutions.drawing_utils

    # 4) Variables for Focus/Pause tracking
    is_focused = False
    focus_start_time = time.time()  # focus start at program start
    focus_start_str = time.strftime("%H:%M:%S", time.localtime(focus_start_time))

    pause_start_time = None
    total_focus_time = 0.0
    total_pause_time = 0.0

    focus_counter = 0
    pause_counter = 0

    # 5) Counters: "drinks" and "phone pickups"
    drinks_count = 0
    phone_pick_count = 0
    is_drinking = False
    is_picking_phone = False

    print("Starting real-time monitoring. Press 'e' or 'q' to end the session.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Couldn't read camera frame.")
                break

            # (a) Object Detection
            frame, detected_objs = detect_objects(frame, net, output_layers, classes_list)

            # (b) Hand Detection
            frame, hand_present, index_coords = detect_hand(frame, hand_model, mp_draw_hands)

            # (c) Face Recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locs = face_recognition.face_locations(rgb_frame)
            face_encs = face_recognition.face_encodings(rgb_frame, face_locs)

            pedro_present = False
            for fe, floc in zip(face_encs, face_locs):
                dist = face_recognition.face_distance([pedro_encoding], fe)[0]
                if dist < 0.6:  # Recognition threshold
                    pedro_present = True
                    top, right, bottom, left = floc
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, "Pedro", (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    break

            # (d) Pose Detection
            pose_detected = detect_pose(frame, pose_model, mp_draw_pose)

            # ------------------------------------------------
            # FOCUS / PAUSE LOGIC WITH DEBOUNCE THRESHOLDS
            # ------------------------------------------------
            if pedro_present or pose_detected:
                focus_counter += 1
                pause_counter = 0
                if focus_counter >= FOCUS_THRESHOLD and not is_focused:
                    # End pause time if we were in pause
                    if pause_start_time is not None:
                        total_pause_time += (time.time() - pause_start_time)
                    is_focused = True
                    print("Transition to Focus.")
            else:
                pause_counter += 1
                focus_counter = 0
                if pause_counter >= PAUSE_THRESHOLD and is_focused:
                    # Accumulate focus time
                    total_focus_time += (time.time() - focus_start_time)
                    is_focused = False
                    pause_start_time = time.time()
                    print("Transition to Pause.")

            # ------------------------------------------------
            # DRINK COUNTING LOGIC
            # ------------------------------------------------
            found_drink = False
            if hand_present and index_coords[0] is not None:
                for (label, (x, y, w, h)) in detected_objs:
                    if label in ["bottle", "cup"]:
                        obj_center = (x + w // 2, y + h // 2)
                        if distance(index_coords, obj_center) <= DRINK_DISTANCE_THRESHOLD:
                            found_drink = True
                            break

            if found_drink and not is_drinking:
                drinks_count += 1
                is_drinking = True
            if not found_drink:
                is_drinking = False

            # ------------------------------------------------
            # PHONE PICKUP LOGIC
            # ------------------------------------------------
            found_phone = False
            if hand_present:
                for (label, _) in detected_objs:
                    if label == "cell phone":
                        found_phone = True
                        break

            if found_phone and not is_picking_phone:
                phone_pick_count += 1
                is_picking_phone = True
            if not found_phone:
                is_picking_phone = False

            # ------------------------------------------------
            # TIME CALCULATION FOR DISPLAY
            # ------------------------------------------------
            now = time.time()
            if is_focused:
                current_focus = total_focus_time + (now - focus_start_time)
                current_pause = total_pause_time
            else:
                current_focus = total_focus_time
                if pause_start_time is not None:
                    current_pause = total_pause_time + (now - pause_start_time)
                else:
                    current_pause = total_pause_time

            # ------------------------------------------------
            # OVERLAY TEXT INFORMATION ON THE FRAME
            # ------------------------------------------------
            cv2.putText(frame, f"Focus Time: {format_time(current_focus)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Pause Time: {format_time(current_pause)}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.putText(frame, f"Focus Start: {focus_start_str}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Beverage Count: {drinks_count}",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.putText(frame, f"Phone Count: {phone_pick_count}",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            cv2.imshow("Work Monitoring", frame)

            # Press 'e' or 'q' to end session
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('e'), ord('q')]:
                print("User requested to end session...")
                break

    except KeyboardInterrupt:
        # If user presses CTRL+C in the terminal
        print("Monitoring ended by user (CTRL+C).")

    except Exception as e:
        print("An error occurred:", e)

    finally:
        # On exit, calculate final focus time if we are currently in focus
        end_time = time.time()
        if is_focused:
            total_focus_time += (end_time - focus_start_time)

        # Create Calendar Event from (focus_start_time) to (end_time)
        # We pass the final current_focus = total_focus_time
        # create_calendar_event(
        #     title="Focus Session",
        #     start_ts=focus_start_time,
        #     end_ts=end_time,
        #     current_focus=total_focus_time
        # ) #todo
        print("Ending session...")

        # Clean up after loop
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
