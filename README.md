# Smart Retail Shelf Monitor

An AI-powered supermarket shelf monitoring application that uses YOLO11n and Flask to detect products in shelf images and estimate product distribution across different shelf sections.

## Project Overview

Smart Retail Shelf Monitor is a computer vision project designed to assist in monitoring supermarket shelves. Users can upload a shelf image through a web application and receive product detection results.

The application identifies visible products, displays bounding boxes, calculates the total number of detected products, and provides a basic shelf-section analysis.

## Features

- **Product Detection:** Detects visible products using a trained YOLO11n model.
- **Product Count:** Displays the total number of detected products.
- **Detection Confidence:** Calculates the average confidence of detected objects.
- **Shelf Section Analysis:** Divides the image into three horizontal sections and estimates potential low-stock areas based on detection counts.
- **Visual Results:** Displays the uploaded image with detection bounding boxes.

## Technologies Used

- Python
- YOLO11n
- Ultralytics
- Flask
- OpenCV
- HTML and CSS
- NumPy

## Model Training and Evaluation

The model was trained using the SKU-110K dataset.

**Training configuration:**
- Model: YOLO11n
- Image size: 640
- Epochs: 5
- Batch size: 8

**Test results:**

| Metric | Score |
|---|---:|
| Precision | 87.4% |
| Recall | 79.1% |
| mAP@50 | 86.8% |
| mAP@50-95 | 51.4% |

## Project Structure

```text
Smart_Retail_Shelf_Monitor/
│
├── app.py
├── requirements.txt
├── models/
│   └── Smart_Retail_Shelf_Monitor_best.pt
│
├── static/
│   ├── uploads/
│   └── results/
│
└── templates/
    └── index.html
```

## How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/Mohitha-7/Smart-Retail-Shelf-Monitor.git
```

2. Open the project folder:

```bash
cd Smart-Retail-Shelf-Monitor
```

3. Create a virtual environment:

```bash
python -m venv venv
```

4. Activate it on Windows:

```powershell
venv\Scripts\Activate.ps1
```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

6. Run the Flask application:

```bash
python app.py
```

7. Open your browser and visit:

```text
http://127.0.0.1:5000
```

## Important Notes

- Product counts represent detected visible objects, not verified inventory quantities.
- Low-stock indicators are heuristic estimates based on the distribution of detections across three image sections.
- The dataset contains generic product-object annotations, so the model does not identify individual brands or SKU names.
- Detection confidence is not the same as overall model accuracy.

## Future Improvements

- Improve shelf-section analysis.
- Add inventory tracking and stock alerts.
- Develop a more detailed monitoring dashboard.
- Explore product classification and SKU-level identification.

## Author

**Mohitha M**

BCA – Artificial Intelligence, Machine Learning and Robotics

[GitHub Profile](https://github.com/Mohitha-7)
