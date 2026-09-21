import torch
import cv2
import torchvision.ops as ops
from pathlib import Path

from model import DroneDetector


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = Path(
    "D:/uav_detector/archive/videos/generate_a_video_of_a_drone_fl.mp4"
)

OUTPUT_DIR = Path(
    "D:\\uav_detector\\deep_cv_model\\results"
)

WEIGHTS_PATH = Path(
    "D:\\uav_detector\\weights\\best_drone_detector.pth"
)

NUM_CLASSES = 5

CLASS_NAMES = [
    "shahed_136",
    "shahed_238",
    "mq9_reaper",
    "dji_mavic",
    "mohajer_6"
]

CONFIDENCE_THRESHOLD = 0.40
IOU_THRESHOLD = 0.30
IMAGE_SIZE = 1280
BOX_SCALE = 0.10


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DRONE DETECTOR")
print("=" * 60)

print(f"Device: {device}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# CHECK MODEL
# ============================================================

if not WEIGHTS_PATH.exists():

    print(
        f"\nERROR: Model weights not found:\n"
        f"{WEIGHTS_PATH}"
    )

    raise SystemExit


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = DroneDetector(
    num_classes=NUM_CLASSES
).to(device)

checkpoint = torch.load(
    WEIGHTS_PATH,
    map_location=device
)

if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )

model.eval()

print("Model loaded successfully!")


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# VIDEO EXTENSIONS
# ============================================================

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv"
}


# ============================================================
# DETECTION FUNCTION
# ============================================================

def detect_frame(frame):

    original_height, original_width = frame.shape[:2]

    resized = cv2.resize(
        frame,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    tensor = (
        torch.from_numpy(rgb)
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(tensor)

    prediction = output[0]

    _, grid_h, grid_w = prediction.shape

    objectness = torch.sigmoid(
        prediction[4]
    )

    mask = (
        objectness >
        CONFIDENCE_THRESHOLD
    )

    ys, xs = torch.where(mask)

    boxes = []
    scores = []
    class_ids = []

    for gy_tensor, gx_tensor in zip(
        ys,
        xs
    ):

        gy = gy_tensor.item()
        gx = gx_tensor.item()

        confidence = objectness[
            gy,
            gx
        ].item()

        bbox = torch.sigmoid(
            prediction[
                0:4,
                gy,
                gx
            ]
        )

        cell_x = bbox[0].item()
        cell_y = bbox[1].item()

        norm_w = bbox[2].item()
        norm_h = bbox[3].item()

        norm_cx = (
            gx + cell_x
        ) / grid_w

        norm_cy = (
            gy + cell_y
        ) / grid_h

        norm_cx = min(
            max(norm_cx, 0.0),
            1.0
        )

        norm_cy = min(
            max(norm_cy, 0.0),
            1.0
        )

        norm_w = min(
            max(norm_w, 0.0),
            1.0
        )

        norm_h = min(
            max(norm_h, 0.0),
            1.0
        )

        center_x = int(
            norm_cx *
            original_width
        )

        center_y = int(
            norm_cy *
            original_height
        )

        box_w = int(
            norm_w *
            original_width *
            BOX_SCALE
        )

        box_h = int(
            norm_h *
            original_height *
            BOX_SCALE
        )

        x1 = max(
            0,
            center_x - box_w // 2
        )

        y1 = max(
            0,
            center_y - box_h // 2
        )

        x2 = min(
            original_width - 1,
            center_x + box_w // 2
        )

        y2 = min(
            original_height - 1,
            center_y + box_h // 2
        )

        if x2 <= x1 or y2 <= y1:
            continue

        class_logits = prediction[
            5:5 + NUM_CLASSES,
            gy,
            gx
        ]

        class_id = torch.argmax(
            class_logits
        ).item()

        boxes.append([
            x1,
            y1,
            x2,
            y2
        ])

        scores.append(
            confidence
        )

        class_ids.append(
            class_id
        )

    # ========================================================
    # NON-MAXIMUM SUPPRESSION
    # ========================================================

    if len(boxes) > 0:

        boxes_tensor = torch.tensor(
            boxes,
            dtype=torch.float32
        )

        scores_tensor = torch.tensor(
            scores,
            dtype=torch.float32
        )

        class_tensor = torch.tensor(
            class_ids,
            dtype=torch.int64
        )

        keep = ops.batched_nms(
            boxes_tensor,
            scores_tensor,
            class_tensor,
            IOU_THRESHOLD
        )

        for index in keep:

            index = index.item()

            x1, y1, x2, y2 = (
                boxes[index]
            )

            confidence = scores[index]

            class_id = class_ids[index]

            if class_id < NUM_CLASSES:

                class_name = (
                    CLASS_NAMES[class_id]
                )

            else:

                class_name = (
                    f"class_{class_id}"
                )

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

            label = (
                f"{class_name} "
                f"{confidence:.2f}"
            )

            text_y = max(
                int(y1) - 10,
                25
            )

            cv2.putText(
                frame,
                label,
                (int(x1), text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    return frame


# ============================================================
# PROCESS SINGLE IMAGE
# ============================================================

def process_image(image_path):

    print(
        f"\nProcessing image: "
        f"{image_path.name}"
    )

    frame = cv2.imread(
        str(image_path)
    )

    if frame is None:

        print(
            "ERROR: Could not read image."
        )

        return

    result = detect_frame(frame)

    output_path = (
        OUTPUT_DIR /
        image_path.name
    )

    cv2.imwrite(
        str(output_path),
        result
    )

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# PROCESS IMAGE FOLDER
# ============================================================

def process_image_folder(folder_path):

    image_files = sorted(
        [
            file
            for file in folder_path.iterdir()
            if file.is_file()
            and file.suffix.lower()
            in IMAGE_EXTENSIONS
        ]
    )

    if len(image_files) == 0:

        print(
            "\nNo images found in folder."
        )

        return

    print(
        f"\nFound {len(image_files)} images."
    )

    print(
        "Starting image detection...\n"
    )

    for number, image_path in enumerate(
        image_files,
        start=1
    ):

        print(
            f"[{number}/{len(image_files)}] "
            f"{image_path.name}"
        )

        process_image(
            image_path
        )

    print(
        "\nAll images processed!"
    )


# ============================================================
# PROCESS VIDEO
# ============================================================

def process_video(video_path):
    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )
    else:
        print("running on CPU")
        

    video_path = Path(video_path)

    print(
        f"\nProcessing video: "
        f"{video_path.name}"
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print(
            "\nERROR: Could not open video."
        )

        return

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    print(
        f"Resolution: "
        f"{width} × {height}"
    )

    print(
        f"FPS: {fps:.2f}"
    )

    print(
        f"Frames: {total_frames}"
    )

    # --------------------------------------------------------
    # Output video
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR /
        f"{video_path.stem}_result.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():

        print(
            "\nERROR: Could not create output video."
        )

        cap.release()

        return

    # --------------------------------------------------------
    # Process frames
    # --------------------------------------------------------

    frame_number = 0

    print(
        "\nStarting video detection...\n"
    )

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        result = detect_frame(
            frame
        )

        cv2.putText(
            result,
            f"Frame: "
            f"{frame_number}/{total_frames}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        writer.write(
            result
        )

        if frame_number % 30 == 0:

            progress = (
                frame_number /
                total_frames *
                100
            )

            print(
                f"Processing: "
                f"{progress:.1f}% "
                f"({frame_number}/{total_frames})"
            )

    cap.release()
    writer.release()

    print(
        "\nVideo detection complete!"
    )

    print(
        f"Output video:\n"
        f"{output_path}"
    )


# ============================================================
# DETERMINE INPUT TYPE
# ============================================================

if __name__ == "__main__":

    if not INPUT_PATH.exists():

        print(
            f"\nERROR: Input does not exist:\n"
            f"{INPUT_PATH}"
        )

        raise SystemExit

    # ========================================================
    # RUN
    # ========================================================

    if INPUT_PATH.is_dir():

        process_image_folder(
            INPUT_PATH
        )

    elif INPUT_PATH.is_file():

        extension = (
            INPUT_PATH.suffix.lower()
        )

        if extension in IMAGE_EXTENSIONS:

            process_image(
                INPUT_PATH
            )

        elif extension in VIDEO_EXTENSIONS:

            process_video(
                INPUT_PATH
            )

        else:

            print(
                f"\nERROR: Unsupported file type:"
                f" {extension}"
            )

    else:

        print(
            "\nERROR: Invalid input path."
        )

    # ========================================================
    # FINISHED
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "DONE"
    )

    print(
        "=" * 60
    )

    print(
        f"Results saved in:\n"
        f"{OUTPUT_DIR}"
    )
