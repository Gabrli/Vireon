# Vireon - Malaria Cell Detection and Classification from Microscopic Images

This project uses a YOLO model to detect and classify different types of malaria cells in microscope images.

![alt text](image-3.png)
![alt text](image-4.png)

## Project Structure

- `main.py` – script for training the YOLO model on your custom dataset.
- `data.yaml` – configuration file with dataset paths and class list.
- `vireon/weights/best.pt` – trained YOLO model weights.
- `vireon/` – folder containing results, plots, and sample images.

## Requirements

- Python 3.8+
- ultralytics (YOLO)
- Data files in YOLO format

Install dependencies:
```sh
pip install ultralytics
```

## Training

Make sure `data.yaml` points to the correct train and validation image folders. Then run:

```sh
python main.py
```

## Configuration

Training parameters can be adjusted in [`main.py`](main.py).

## Results

The [`vireon/`](vireon/) folder contains:
- plots (`F1_curve.png`, `PR_curve.png`, etc.)
- confusion matrices (`confusion_matrix.png`)
- sample images with predictions

## License

For personal, educational, and research use.