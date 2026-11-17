
from flask import Flask, render_template, request
from ultralytics import YOLO
from pathlib import Path
from werkzeug.utils import secure_filename
import cv2
import uuid

app = Flask(__name__)

# Project directories
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "Smart_Retail_Shelf_Monitor_best.pt"
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
RESULT_FOLDER = BASE_DIR / "static" / "results"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

# Load trained YOLO model
model = YOLO(str(MODEL_PATH))


@app.route("/", methods=["GET", "POST"])
def index():
    result_image = None
    product_count = None
    confidence = None
    error = None
    zone_report = []

    if request.method == "POST":

        if "image" not in request.files:
            error = "Please select an image."

        else:
            file = request.files["image"]

            if file.filename == "":
                error = "No image selected."

            elif not file.filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                error = "Please upload a JPG or PNG image."

            else:
                filename = secure_filename(file.filename)
                unique_name = f"{uuid.uuid4().hex}_{filename}"

                image_path = UPLOAD_FOLDER / unique_name
                file.save(str(image_path))

                # Run YOLO product detection
                results = model.predict(
                    source=str(image_path),
                    conf=0.25,
                    imgsz=640,
                    max_det=718,
                    verbose=False
                )

                result = results[0]
                product_count = len(result.boxes)

                # Divide image into 3 horizontal zones
                image_height = result.orig_shape[0]
                zone_counts = [0, 0, 0]

                for box in result.boxes:
                    center_y = float(box.xywh[0][1].item())

                    zone_index = min(
                        int((center_y / image_height) * 3),
                        2
                    )

                    zone_counts[zone_index] += 1

                # Estimate low-stock zones
                average_count = product_count / 3 if product_count > 0 else 0

                for i, count in enumerate(zone_counts):
                    if average_count > 0 and count < average_count * 0.5:
                        status = "Potential Low Stock"
                    else:
                        status = "Normal"

                    zone_report.append({
                        "name": f"Shelf Section {i + 1}",
                        "count": count,
                        "status": status
                    })

                # Calculate average detection confidence
                if product_count > 0:
                    confidence = round(
                        float(result.boxes.conf.mean().item()) * 100,
                        2
                    )
                else:
                    confidence = 0

                # Draw detection boxes
                annotated_image = result.plot(
                    labels=False,
                    conf=False,
                    line_width=2
                )

                result_path = RESULT_FOLDER / f"result_{unique_name}"

                cv2.imwrite(
                    str(result_path),
                    annotated_image
                )

                result_image = f"results/result_{unique_name}"

    return render_template(
        "index.html",
        result_image=result_image,
        product_count=product_count,
        confidence=confidence,
        zone_report=zone_report,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
