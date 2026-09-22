# 🧠 YuNet Face Detection & Privacy Blur using OpenCV

## 📌 Project Overview

This project demonstrates **Face Detection using YuNet**, a lightweight
face detection model provided through the **OpenCV Zoo** ecosystem.

The practical covers:

-   Loading the YuNet ONNX face detection model.
-   Detecting faces in static images.
-   Drawing face bounding boxes.
-   Displaying five facial landmarks.
-   Displaying detection confidence scores.
-   Running a score-threshold experiment.
-   Comparing detection results at multiple thresholds.
-   Applying Gaussian blur to detected face regions.
-   Creating a privacy-preserving anonymised image.

The complete practical is implemented in a **Jupyter Notebook** using
Python, OpenCV and Matplotlib.


## 📺 Presentation Video

👉 **https://drive.google.com/file/d/16ybJtslHuO4M240rghvHcEmVf1IjCBSL/view?usp=sharing**

------------------------------------------------------------------------

# 🎯 Objectives

-   Understand the basics of face detection.
-   Download and load the YuNet ONNX model.
-   Store the model in a structured project directory.
-   Load static face images.
-   Detect faces using YuNet.
-   Draw bounding boxes around detected faces.
-   Display five facial landmarks.
-   Display the detection confidence score.
-   Experiment with different score thresholds.
-   Compare thresholds `0.5`, `0.7` and `0.9`.
-   Count detected faces for each threshold.
-   Apply Gaussian blur to detected face regions.
-   Create a privacy-preserving anonymised image.
-   Save detection and privacy visualizations.
-   Understand the difference between face detection and face
    recognition.
-   Document the complete practical in a GitHub-ready repository.

------------------------------------------------------------------------

# 📂 Project Input

## Face Images

The practical works with static face photographs stored inside:

``` text
data/images/
```

Recommended filenames:

``` text
face_01.jpg
face_02.jpg
face_03.jpg
face_04.jpg
```

The notebook automatically searches for supported image formats:

``` text
.jpg
.jpeg
.png
.webp
```

and filenames containing:

``` text
face
```

------------------------------------------------------------------------

# 🤖 YuNet Model

The model used in this project is:

``` text
face_detection_yunet_2023mar.onnx
```

YuNet is a lightweight face detection model suitable for efficient face
detection.

### Official OpenCV Zoo Model URL

``` text
https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

The downloaded model is stored at:

``` text
data/models/face_detection_yunet_2023mar.onnx
```

------------------------------------------------------------------------

# 🚀 Project Workflow

``` text
Install Dependencies
        ↓
Create Project Folders
        ↓
Download YuNet ONNX Model
        ↓
Load Face Images
        ↓
Create YuNet Detector
        ↓
Detect Faces
        ↓
Draw Bounding Boxes
        ↓
Draw Facial Landmarks
        ↓
Display Confidence Scores
        ↓
Run Threshold Experiment
        ↓
Compare 0.5 / 0.7 / 0.9
        ↓
Detect Face Region
        ↓
Apply Gaussian Blur
        ↓
Save Anonymised Image
        ↓
Display Before / After
```

------------------------------------------------------------------------

# 📁 Project Folder Structure

``` text
YuNet_Face_Detection/
│
├── data/
│   ├── images/
│   │   ├── face_01.jpg
│   │   ├── face_02.jpg
│   │   ├── face_03.jpg
│   │   ├── face_04.jpg
│   │   └── face_privacy_blurred.jpg
│   │
│   └── models/
│       └── face_detection_yunet_2023mar.onnx
│
├── plots/
│   ├── face_detection_comparison.png
│   └── face_privacy_blur.png
│
├── YuNet_Face_Detection.ipynb
├── README.md
└── requirements.txt
```

------------------------------------------------------------------------

# 🛠️ Technologies Used

-   Python
-   OpenCV
-   YuNet
-   ONNX
-   NumPy
-   Matplotlib
-   Jupyter Notebook
-   urllib
-   pathlib

------------------------------------------------------------------------

# 📦 Installation

Install the required Python packages:

``` bash
pip install --upgrade opencv-python matplotlib numpy
```

For Jupyter Notebook:

``` bash
pip install notebook
```

Or inside a Jupyter cell:

``` python
!pip install --upgrade opencv-python matplotlib numpy
```

After upgrading OpenCV, restart the Jupyter kernel.

------------------------------------------------------------------------

# 📖 Step 3.1 -- Download / Load YuNet

## Create Required Folders

``` python
from pathlib import Path

BASE_DIR = Path.cwd()

DATA = BASE_DIR / "data" / "images"
MODELS = BASE_DIR / "data" / "models"
PLOTS = BASE_DIR / "plots"

DATA.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)
```

------------------------------------------------------------------------

## Define YuNet Model Path

``` python
yunet_path = (
    MODELS /
    "face_detection_yunet_2023mar.onnx"
)
```

------------------------------------------------------------------------

## Define Official Model URL

``` python
yunet_url = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
```

------------------------------------------------------------------------

## Download the Model

``` python
import urllib.request

if not yunet_path.exists():

    print("Downloading YuNet model...")

    try:

        urllib.request.urlretrieve(
            yunet_url,
            str(yunet_path)
        )

        print(
            "YuNet model downloaded successfully."
        )

    except Exception as e:

        raise RuntimeError(
            f"Could not download YuNet model: {e}"
        )

else:

    print(
        "YuNet model already exists."
    )
```

------------------------------------------------------------------------

# 🔍 Create YuNet Detector

``` python
import cv2

def create_yunet(img):

    if img is None:
        raise ValueError(
            "Input image is None."
        )

    h, w = img.shape[:2]

    detector = cv2.FaceDetectorYN_create(
        str(yunet_path),
        "",
        (w, h),
        0.8,
        0.3,
        5000
    )

    detector.setInputSize(
        (w, h)
    )

    return detector
```

### Detector Parameters

  Parameter      Meaning
  -------------- ------------------------------
  `yunet_path`   Path to YuNet ONNX model
  `""`           Model configuration
  `(w, h)`       Input image size
  `0.8`          Score/confidence threshold
  `0.3`          NMS threshold
  `5000`         Maximum number of detections

------------------------------------------------------------------------

# 📷 Step 3.2 -- Detect Faces on Static Real Images

The notebook searches the image directory automatically.

``` python
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}

face_files = sorted([
    p for p in DATA.iterdir()
    if p.is_file()
    and p.suffix.lower()
    in IMAGE_EXTENSIONS
    and "face"
    in p.stem.lower()
])
```

If no images are found:

``` text
No face images found.
```

the images should be placed inside:

``` text
data/images/
```

------------------------------------------------------------------------

# 🖼️ Face Detection

``` python
import matplotlib.pyplot as plt

if not face_files:

    print(
        "No face images found."
    )

else:

    fig, axes = plt.subplots(
        len(face_files),
        1,
        figsize=(
            10,
            5 * len(face_files)
        )
    )

    if len(face_files) == 1:
        axes = [axes]

    for ax, image_path in zip(
        axes,
        face_files
    ):

        img = cv2.imread(
            str(image_path)
        )

        if img is None:

            print(
                "Could not read:",
                image_path
            )

            ax.axis("off")
            continue

        detector = create_yunet(
            img
        )

        _, faces = detector.detect(
            img
        )

        output = img.copy()

        face_count = 0

        if faces is not None:

            face_count = len(faces)

            for face in faces:

                x, y, w, h = (
                    face[:4]
                    .astype(int)
                )

                score = float(
                    face[14]
                )

                # Bounding box
                cv2.rectangle(
                    output,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                # Five facial landmarks
                for j in range(5):

                    lx = int(
                        face[
                            4 + 2 * j
                        ]
                    )

                    ly = int(
                        face[
                            5 + 2 * j
                        ]
                    )

                    cv2.circle(
                        output,
                        (lx, ly),
                        4,
                        (255, 0, 0),
                        -1
                    )

                # Confidence score
                cv2.putText(
                    output,
                    f"Face: {score:.2f}",
                    (
                        x,
                        max(
                            25,
                            y - 10
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        ax.imshow(
            cv2.cvtColor(
                output,
                cv2.COLOR_BGR2RGB
            )
        )

        ax.set_title(
            f"{image_path.name} | "
            f"Faces detected: {face_count}"
        )

        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        PLOTS /
        "face_detection_comparison.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()
```

------------------------------------------------------------------------

# 👁️ YuNet Detection Output

For every detected face, YuNet provides:

``` text
x
y
width
height
```

followed by five facial landmarks and a detection score.

The approximate structure is:

``` text
[
    x, y, width, height,
    eye1_x, eye1_y,
    eye2_x, eye2_y,
    nose_x, nose_y,
    mouth1_x, mouth1_y,
    mouth2_x, mouth2_y,
    score
]
```

------------------------------------------------------------------------

# 📦 Bounding Box

The bounding box represents the location of the face.

``` python
x, y, w, h = face[:4].astype(int)
```

The rectangle is drawn using:

``` python
cv2.rectangle(
    output,
    (x, y),
    (x + w, y + h),
    (0, 255, 0),
    2
)
```

------------------------------------------------------------------------

# 📍 Facial Landmarks

YuNet provides five facial landmark points.

They represent important facial locations such as:

-   Eye region
-   Eye region
-   Nose
-   Mouth corner
-   Mouth corner

The landmarks are displayed using blue circles.

``` python
for j in range(5):

    lx = int(face[4 + 2 * j])
    ly = int(face[5 + 2 * j])

    cv2.circle(
        output,
        (lx, ly),
        4,
        (255, 0, 0),
        -1
    )
```

------------------------------------------------------------------------

# 🎯 Detection Confidence Score

The detection score indicates the confidence associated with a detected
face.

``` python
score = float(face[14])
```

It is displayed above the bounding box:

``` text
Face: 0.98
```

The exact value depends on the input image and detection result.

------------------------------------------------------------------------

# 📊 Detection Output

The detection visualization is saved as:

``` text
plots/face_detection_comparison.png
```

The output contains:

-   Face bounding boxes
-   Facial landmarks
-   Confidence scores
-   Image filenames
-   Number of detected faces

------------------------------------------------------------------------

# 🧪 Step 3.4 -- Score Threshold Experiment

The threshold experiment tests:

``` text
0.5
0.7
0.9
```

A score threshold determines the minimum confidence required for a face
detection to be accepted.

------------------------------------------------------------------------

# ⚙️ Threshold Detection Function

``` python
def detect_faces_threshold(
    img,
    threshold
):

    if img is None:

        raise ValueError(
            "Input image is empty."
        )

    h, w = img.shape[:2]

    detector = cv2.FaceDetectorYN_create(
        str(yunet_path),
        "",
        (w, h),
        threshold,
        0.3,
        5000
    )

    detector.setInputSize(
        (w, h)
    )

    _, faces = detector.detect(
        img
    )

    if faces is None:
        return []

    return faces
```

------------------------------------------------------------------------

# 🔢 Run Threshold Experiment

``` python
thresholds = [
    0.5,
    0.7,
    0.9
]

counts = {}

img = cv2.imread(
    str(face_files[0])
)

if img is None:

    raise ValueError(
        "Unable to read experiment image."
    )

for threshold in thresholds:

    faces = detect_faces_threshold(
        img,
        threshold
    )

    counts[threshold] = len(
        faces
    )

    print(
        f"Threshold {threshold}: "
        f"{len(faces)} face(s)"
    )
```

------------------------------------------------------------------------

# 📊 Threshold Comparison

The output is generated from the actual image and should not be
hard-coded.

Example format:

    Threshold   Faces Detected
  ----------- ----------------
          0.5    Actual result
          0.7    Actual result
          0.9    Actual result

### General Interpretation

    Threshold Behaviour
  ----------- ---------------------------------
        `0.5` Lower confidence requirement
        `0.7` Moderate confidence requirement
        `0.9` Higher confidence requirement

A lower threshold can accept weaker detections.

A higher threshold requires stronger confidence.

The exact detection count depends on the image.

------------------------------------------------------------------------

# 🔐 Privacy Blur Experiment

The second part of the experiment uses detected face regions to perform
privacy anonymisation.

The workflow is:

``` text
Image
  ↓
YuNet Face Detection
  ↓
Face Bounding Box
  ↓
Crop Face Region
  ↓
Gaussian Blur
  ↓
Replace Face Region
  ↓
Anonymised Image
```

------------------------------------------------------------------------

# 🌫️ Gaussian Blur

Gaussian Blur is applied only to the detected face region.

``` python
blurred = img.copy()

blur_threshold = 0.7

faces = detect_faces_threshold(
    img,
    blur_threshold
)

for face in faces:

    x, y, w, h = (
        face[:4]
        .astype(int)
    )

    x1 = max(
        0,
        x
    )

    y1 = max(
        0,
        y
    )

    x2 = min(
        img.shape[1],
        x + w
    )

    y2 = min(
        img.shape[0],
        y + h
    )

    if (
        x2 > x1
        and y2 > y1
    ):

        roi = blurred[
            y1:y2,
            x1:x2
        ]

        if (
            roi.shape[0] >= 3
            and roi.shape[1] >= 3
        ):

            blurred_roi = (
                cv2.GaussianBlur(
                    roi,
                    (31, 31),
                    0
                )
            )

            blurred[
                y1:y2,
                x1:x2
            ] = blurred_roi
```

------------------------------------------------------------------------

# 💾 Save Anonymised Image

``` python
blur_output_path = (
    DATA /
    "face_privacy_blurred.jpg"
)

cv2.imwrite(
    str(blur_output_path),
    blurred
)

print(
    "Blurred image saved to:",
    blur_output_path
)
```

------------------------------------------------------------------------

# 🖼️ Before and After Visualization

``` python
fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 6)
)

axes[0].imshow(
    cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )
)

axes[0].set_title(
    "Before Anonymisation"
)

axes[0].axis("off")


axes[1].imshow(
    cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2RGB
    )
)

axes[1].set_title(
    "Face-region Gaussian Blur"
)

axes[1].axis("off")


plt.tight_layout()

plt.savefig(
    PLOTS /
    "face_privacy_blur.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()
```

------------------------------------------------------------------------

# 📊 Final Outputs

## Face Detection Output

``` text
plots/face_detection_comparison.png
```

This image contains:

-   Detected face
-   Bounding box
-   Five facial landmarks
-   Detection score

------------------------------------------------------------------------

## Privacy Blur Output

``` text
plots/face_privacy_blur.png
```

This image contains:

``` text
Before Anonymisation
        vs
Face-region Gaussian Blur
```

------------------------------------------------------------------------

## Blurred Image

``` text
data/images/face_privacy_blurred.jpg
```

This is the saved anonymised image.

------------------------------------------------------------------------

# 🧠 Face Detection vs Face Recognition

These two concepts are different.

### Face Detection

Answers:

``` text
Where is a face?
```

The output is generally a bounding box and confidence score.

### Face Recognition

Answers:

``` text
Whose face is this?
```

Recognition attempts to identify or verify an individual.

### This Project

This project performs:

``` text
FACE DETECTION
```

It does **not** perform identity recognition.

------------------------------------------------------------------------

# 🔍 Understanding Important Parameters

## Score Threshold

``` python
threshold
```

Controls the minimum confidence required for accepting a detection.

------------------------------------------------------------------------

## NMS Threshold

``` python
0.3
```

NMS means:

``` text
Non-Maximum Suppression
```

It helps remove overlapping duplicate detections.

------------------------------------------------------------------------

## Top-K

``` python
5000
```

Controls the maximum number of candidate detections considered.

------------------------------------------------------------------------

## Input Size

``` python
(w, h)
```

The detector input size is set to the actual image dimensions.

``` python
detector.setInputSize(
    (w, h)
)
```

------------------------------------------------------------------------

# 🎨 Why BGR is Converted to RGB

OpenCV loads images in:

``` text
BGR
```

Matplotlib expects:

``` text
RGB
```

Therefore:

``` python
cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)
```

is used before displaying OpenCV images with Matplotlib.

------------------------------------------------------------------------

# 🌫️ Why Gaussian Blur?

Gaussian Blur removes or reduces fine visual details.

For privacy applications, blurring the detected face region helps reduce
visible facial detail while retaining the rest of the image.

The project uses:

``` python
(31, 31)
```

as the Gaussian kernel.

The kernel dimensions are odd numbers, as required by OpenCV for this
operation.

------------------------------------------------------------------------

# ⚠️ Limitations

The project has several limitations:

-   Face detection depends on image quality.
-   Very small faces can be difficult to detect.
-   Extreme face angles can affect detection.
-   Poor lighting can reduce detection quality.
-   Occlusion can affect detection.
-   Different thresholds produce different detection results.
-   A very high threshold may reject difficult detections.
-   A very low threshold may accept weaker detections.
-   Gaussian blur is an anonymisation technique and is not a guarantee
    of complete privacy.
-   Face detection does not identify a person's identity.

------------------------------------------------------------------------

# 🚀 Future Scope

Possible extensions include:

-   Real-time webcam face detection.
-   Video face detection.
-   Automatic video face anonymisation.
-   Face tracking.
-   FPS/performance measurement.
-   Threshold optimization.
-   Comparison with other face detectors.
-   Detection evaluation using precision and recall.
-   Automatic privacy masking.
-   Real-time privacy filtering.
-   Deployment as a web application.
-   Integration with CCTV/video-processing pipelines.

------------------------------------------------------------------------

# 🧪 Troubleshooting

## Error: `No face images found`

Check:

``` text
data/images/
```

Make sure filenames contain:

``` text
face
```

Example:

``` text
face_01.jpg
face_02.jpg
face_03.jpg
face_04.jpg
```

------------------------------------------------------------------------

## Error: YuNet model not found

Make sure this file exists:

``` text
data/models/face_detection_yunet_2023mar.onnx
```

The notebook automatically downloads it if it is missing.

------------------------------------------------------------------------

## Error: `cv2` not found

Run:

``` bash
pip install --upgrade opencv-python
```

------------------------------------------------------------------------

## Error: `matplotlib` not found

Run:

``` bash
pip install matplotlib
```

------------------------------------------------------------------------

## Error: Image is `None`

Check:

-   Image path
-   File extension
-   File existence
-   Image file integrity

------------------------------------------------------------------------

## Error in `FaceDetectorYN_create`

Use the positional-argument form:

``` python
cv2.FaceDetectorYN_create(
    str(yunet_path),
    "",
    (w, h),
    threshold,
    0.3,
    5000
)
```

This avoids compatibility problems caused by unsupported keyword
arguments in some OpenCV versions.

------------------------------------------------------------------------

## Important Syntax Error

Incorrect:

``` python
detector.setInputSize((w, h))w
```

Correct:

``` python
detector.setInputSize((w, h))
```

------------------------------------------------------------------------

# ❓ Viva Questions & Answers

## Q1. What is YuNet?

YuNet is a lightweight face detection model used to detect faces in
images and video.

## Q2. What is OpenCV?

OpenCV is an open-source computer-vision library used for image and
video processing.

## Q3. What is face detection?

Face detection identifies the location of faces in an image or video.

## Q4. What is a bounding box?

A bounding box is a rectangle that identifies the location and size of a
detected face.

## Q5. What are facial landmarks?

Facial landmarks are important points on a detected face. YuNet provides
five landmark locations.

## Q6. What is an ONNX model?

ONNX stands for Open Neural Network Exchange. It is a standardized model
format that allows machine-learning models to be used across different
frameworks and runtimes.

## Q7. What is the score threshold?

It is the minimum confidence required for a detection to be accepted.

## Q8. What happens when the threshold increases?

The detector becomes more selective because a detection must have a
higher score to be accepted.

## Q9. What happens when the threshold decreases?

Lower-confidence detections can be accepted.

## Q10. What is NMS?

NMS means Non-Maximum Suppression. It helps remove overlapping duplicate
detections.

## Q11. Why are landmarks displayed?

Landmarks provide important facial reference points and demonstrate
additional information returned by the detector.

## Q12. Why do we use Gaussian Blur?

Gaussian Blur is used to reduce facial details for privacy
anonymisation.

## Q13. Why is `(31,31)` used?

It provides strong smoothing of the face region. The kernel dimensions
are odd numbers.

## Q14. Is this face recognition?

No. This project performs face detection, not identity recognition.

## Q15. Why do we test multiple thresholds?

To observe how the detector's confidence requirement affects detection
results.

## Q16. Why do we use `cv2.cvtColor()`?

OpenCV uses BGR channel ordering while Matplotlib expects RGB.

## Q17. Why do we check `faces is None`?

YuNet can return `None` when no face is detected.

## Q18. What does `face[:4]` contain?

It contains:

``` text
x
y
width
height
```

## Q19. What does `face[14]` represent?

It represents the detection confidence score.

## Q20. What is privacy anonymisation?

It is the process of modifying identifiable information so that
sensitive visual details are less recognizable.

------------------------------------------------------------------------

# 📝 Practical Explanation for Viva

A concise explanation of the complete practical is:

> "In this practical, I used YuNet, a lightweight face detection model,
> with OpenCV. First, I downloaded the YuNet ONNX model and stored it in
> the models directory. Then I loaded static face images and detected
> faces using the YuNet detector. For every detected face, I displayed a
> bounding box, five facial landmarks and the confidence score. Next, I
> performed a threshold experiment using 0.5, 0.7 and 0.9 to observe the
> effect of detection confidence. Finally, I applied Gaussian blur to
> the detected face region to demonstrate a privacy-preserving
> anonymisation technique. The detection and privacy results were
> visualized and saved as output plots."

------------------------------------------------------------------------

# 📌 Key Learnings

-   Computer Vision
-   Face Detection
-   YuNet
-   OpenCV
-   ONNX
-   Bounding Boxes
-   Facial Landmarks
-   Confidence Scores
-   Score Thresholds
-   Non-Maximum Suppression
-   Gaussian Blur
-   Privacy Anonymisation
-   Image Processing
-   Matplotlib Visualization
-   Jupyter Notebook
-   Model Loading
-   Static Image Processing

------------------------------------------------------------------------

# 📤 Submission Checklist

Before submitting the practical, verify:

-   [ ] Python is installed.
-   [ ] OpenCV is installed.
-   [ ] NumPy is installed.
-   [ ] Matplotlib is installed.
-   [ ] YuNet ONNX model is available.
-   [ ] `data/images/` exists.
-   [ ] Face images are available.
-   [ ] `data/models/` contains the YuNet model.
-   [ ] YuNet loads without errors.
-   [ ] Static face detection runs successfully.
-   [ ] Bounding boxes are visible.
-   [ ] Five landmarks are visible.
-   [ ] Confidence scores are displayed.
-   [ ] Thresholds 0.5, 0.7 and 0.9 are tested.
-   [ ] Face counts are printed.
-   [ ] Privacy blur runs successfully.
-   [ ] Blurred image is saved.
-   [ ] Detection comparison plot is saved.
-   [ ] Privacy comparison plot is saved.
-   [ ] Notebook runs from beginning to end without errors.
-   [ ] README.md is included.

------------------------------------------------------------------------

# 📁 Final Repository Structure

``` text
YuNet_Face_Detection/
│
├── data/
│   ├── images/
│   │   ├── face_01.jpg
│   │   ├── face_02.jpg
│   │   ├── face_03.jpg
│   │   ├── face_04.jpg
│   │   └── face_privacy_blurred.jpg
│   │
│   └── models/
│       └── face_detection_yunet_2023mar.onnx
│
├── plots/
│   ├── face_detection_comparison.png
│   └── face_privacy_blur.png
│
├── YuNet_Face_Detection.ipynb
├── README.md
└── requirements.txt
```

------------------------------------------------------------------------

# ▶️ How to Run

Open the project in Jupyter Notebook or VS Code.

Start Jupyter:

``` bash
jupyter notebook
```

Open:

``` text
YuNet_Face_Detection.ipynb
```

Run all cells sequentially.

Before final submission:

``` text
Kernel → Restart & Run All
```

Confirm that the complete notebook executes without errors.

------------------------------------------------------------------------

# 📦 requirements.txt

Recommended contents:

``` text
opencv-python
numpy
matplotlib
jupyter
```

Install everything with:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 🏁 Conclusion

This practical demonstrates a complete face-detection and
privacy-anonymisation workflow using YuNet and OpenCV.

The project starts with model loading and static-image face detection,
then extends the experiment by studying different score thresholds.
Finally, detected face regions are blurred using Gaussian Blur to
demonstrate a practical privacy-preserving computer-vision application.

The practical provides hands-on understanding of:

``` text
YuNet
   ↓
Face Detection
   ↓
Bounding Boxes
   ↓
Facial Landmarks
   ↓
Confidence Scores
   ↓
Threshold Experiment
   ↓
Privacy Blur
```

------------------------------------------------------------------------

# 👩‍💻 Author

**Sanika Kale**

MCA Student \| AI/ML & Data Science

Red & White Skill Education

------------------------------------------------------------------------

⭐ **If you found this project useful, consider giving it a star on
GitHub!**
