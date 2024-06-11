from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
import numpy as np

# Paths
path_weights_food = "/Users/gustavo/Documents/Studies/Unicamp/S8/mc906/projeto-mc906/models/final/food.pt"
path_weights_plate = "/Users/gustavo/Documents/Studies/Unicamp/S8/mc906/projeto-mc906/models/final/plate.pt"
path_img = "/Users/gustavo/Documents/Studies/Unicamp/S8/mc906/projeto-mc906/app/main/image_version/images/bandeco.jpg"

# Load density map (kg/m^3)
density_map = {
    "apple": 750,
    "apricots": 750,
    "avocado": 634,
    "baked_goods": 415,
    "banana": 634,
    "berries": 626,
    "beverage": 1000,
    "biscuit": 470,
    "blueberries": 626,
    "bread": 290,
    "butter": 911,
    "cherries": 630,
    "chicken": 896,
    "coca": 1031,
    "coconut": 352,
    "coffee": 1002,
    "condiments": 0,
    "cheese": 340,
    "crepe": 571,
    "curry": 426,
    "dates": 0,
    "desserts": 750,
    "egg": 600,
    "falafel": 676,
    "figs": 750,
    "fish": 593,
    "flakes": 173,
    "fondue": 909,
    "french_fries": 240,
    "grain": 782,
    "grapefruit": 761,
    "grapes": 638,
    "honey": 1433,
    "juice": 1054,
    "kaki": 0,
    "kiwi": 748,
    "kefir_drink": 1000,
    "legumes": 642,
    "meat": 1037,
    "lemon": 761,
    "mandarine": 761,
    "mango": 676,
    "melon": 985,
    "milk_coffee": 1107,
    "mixed_milk": 1107,
    "muesli": 359,
    "nectarine": 604,
    "nuts": 558,
    "olives": 568,
    "omelette": 1014,
    "orange": 761,
    "pasta": 507,
    "peach": 604,
    "pear": 750,
    "pineapple": 697,
    "pistachio": 520,
    "pizza": 503,
    "plum": 697,
    "polenta": 704,
    "pork": 566,
    "pomegranate": 592,
    "porridge": 730,
    "potato": 562,
    "raisins_dried": 676,
    "raspberries": 520,
    "rice": 693,
    "risotto": 879,
    "salad": 334,
    "sandwich": 1035,
    "sausage": 583,
    "seeds": 609,
    "shrimp": 541,
    "soft_drink": 1050,
    "soups": 1036,
    "spreads": 1014,
    "strawberries": 642,
    "sweets": 750,
    "tea": 1000,
    "tofu": 1065,
    "vegetables": 642,
    "water": 1000,
    "watermelon": 642,
    "wine": 981,
    "yogurt": 1035,
}
categories = list(density_map.keys())

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
plate_h = 1e-2
plate_d = 30e-2

# get pixels to meters conversion rate
_x, _y, w, h = result_plate.boxes[0].xywh[0].tolist()  # assuming one plate
plate_d_pixels = (w + h) / 2
meters_per_pixel = plate_d / plate_d_pixels

# Process results list
print("Detected:")
for food_box in result_food.boxes:
    cl = int(food_box.cls[0].tolist())
    _x, _y, w, h = food_box.xywh[0].tolist()
    area = meters_per_pixel**2 * np.pi * w * h / 4  # ellipse (m^2)
    volume = plate_h * area  # m^3
    density = density_map[categories[cl]]  # kg/m^3
    weight = density * volume * 1000  # g
    print(f"  * {weight:.1f} g of {categories[cl]}")
