from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
from pathlib import Path
import numpy as np
import json

# Paths
path_weights_food = "best_food.pt"
path_weights_plate = "best_plate.pt"
path_img = "oi.jpeg"
path_density_map = "food_density.json"

# Load density map (kg/m^3)
density_map = json.loads(Path(path_density_map).read_text())

# Load models
model_food = YOLO(path_weights_food)
model_plate = YOLO(path_weights_plate)

# Run batched inference on a list of images
result_food: Results = model_food([path_img])[0]  # access to index 0 because only 1 image
result_plate: Results = model_plate([path_img])[0]  # access to index 0 because only 1 image

# validation
if not isinstance(result_plate.boxes, Boxes):
    raise RuntimeError("no plate boxes returned")
if not isinstance(result_food.boxes, Boxes):
    raise RuntimeError("no food boxes returned")


# TODO: ask the user for these values
# IDEA: one value for shallow, medium and deep plates
plate_h = 0.1
plate_d = 0.24

# get pixels to meters conversion rate
_x, _y, w, h = result_plate.boxes[0].xywh[0].tolist()  # assuming one plate
plate_d_pixels = (w + h) / 2
meters_per_pixel = plate_d / plate_d_pixels

# Process results list
for food_box in result_food.boxes:
    cl = int(food_box.cls[0].tolist()[0])
    _x, _y, w, h = food_box.xywh[0].tolist()
    area = meters_per_pixel * np.pi * w * h / 4  # ellipse (m^2)
    volume = plate_h * area  # m^3
    density = density_map[cl]  # kg/m^3
    weight = density * volume  # kg
