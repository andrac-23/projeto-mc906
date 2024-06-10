import os
import tkinter as tk

import cv2 as cv
import numpy as np
import requests
from dotenv import load_dotenv
from typing import TypedDict
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

DENSITY_MAP = {
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

ResultsFood = list[tuple[str, Boxes]]
ResultsPlate = Boxes
class Results(TypedDict):
    food: ResultsFood
    plate: ResultsPlate

def get_yolo_detection_results(frame, models_dir) -> Results:
    print("Realizando detecção de objetos na imagem atual...")
    food_model = YOLO(build_path(models_dir,'food.pt'))
    f_results = food_model.predict([frame])[0]
    if not isinstance(f_results.boxes, Boxes):
        raise RuntimeError("no plate boxes returned")

    plate_model = YOLO(build_path(models_dir,'plate.pt'))
    p_results = plate_model.predict([frame])[0]
    if not isinstance(p_results.boxes, Boxes):
        raise RuntimeError("no food boxes returned")

    return Results(
        food=[
            (f_results.names[int(cls)], bbox)
            for cls, bbox in zip(f_results.boxes.cls, f_results.boxes)
        ],
        plate=p_results.boxes[0],
    )

def insert_food_regions_detected(frame, yolov8_results: Results):
    # forma do retangulo
    frame_height, frame_width = frame.shape[:2]
    top_left_point = (int(0.87 * frame_width), int(0.075 * frame_height))
    bottom_right_point = (int(0.97 * frame_width), int(0.925 * frame_height))

    rect_width = bottom_right_point[0] - top_left_point[0]
    rect_height = bottom_right_point[1] - top_left_point[1]

    # p_result, f_results = yolov8_results[0], yolov8_results[1]
    percentual_healthy, calories_density = __healthy_analisys(yolov8_results)

    # preencher retangulo
    fill_percentage = percentual_healthy/100  # 60% fill
    fill_height = int(rect_height * fill_percentage)

    # Define the filled rectangle parameters starting from the base
    fill_start_point = (top_left_point[0], bottom_right_point[1] - fill_height)
    fill_end_point = bottom_right_point
    cv.rectangle(frame, fill_start_point, fill_end_point, (52, 57, 244), cv.FILLED)

    # texto na base do retangulo
    text = "{:.2f}".format(percentual_healthy)
    font_scale = 1
    font_thickness = 1
    (text_width, text_height), baseline = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    x = top_left_point[0] + (rect_width - text_width) // 2
    y = bottom_right_point[1] - text_height - 50
    cv.putText(frame, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    cv.rectangle(frame, top_left_point, bottom_right_point, (8, 184, 27), 2)

    font_scale_f = 0.5
    for name, box in yolov8_results['food']:
        xmin, ymin, xmax, ymax = list(map(int, box.xyxy[0]))

        cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        text_x = xmin
        text_y = ymin - 10
        text_y = max(text_y, 10)
        cv.putText(frame, name, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, font_scale_f, (255, 0, 0), 1)

    return frame

def build_path(*paths):
    return os.path.join(*paths)

def __insert_title_to_frame(frame, text):
    font_scale = 1
    font_thickness = 2
    (text_width, text_height), baseline = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    height, width = frame.shape[:2]
    x = (width - text_width) // 2 # centralizado
    y = int(0.075 * height) # 7.5% da altura
    cv.putText(frame, text, (x, y), cv.FONT_HERSHEY_COMPLEX, font_scale, (255,255,255), font_thickness)
    return

def __get_screen_resolution():
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    return screen_width, screen_height

def __move_window_to_center(window_name, width, height):
    screen_width, screen_height = __get_screen_resolution()
    screen_height = int(screen_height * 0.50) # 90% da tela
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    cv.moveWindow(window_name, x, y)
    return

def __get_food_on_api(food_name):
    root_dir = os.path.abspath(build_path(os.path.dirname(__file__), '..', '..'))
    load_dotenv(build_path(root_dir, '.env'))

    api_url = str(os.getenv('API_URL'))
    api_key = os.getenv('API_KEY')

    params = {
        'query': str(food_name),
        'dataType': 'Branded',
        'pageSize': '1',
        'pageNumber': '0',
        'sortBy': 'dataType.keyword',
        'sortOrder': 'asc',
        'api_key': api_key
    }
    response = requests.get(api_url, params=params)

    if response.status_code == 200:
        return response.json()["foods"][0]
    else:
        raise Exception(f"Erro na requisição: {response.status_code}")

def __healthy_analisys(yolov8_results: Results, plate_h=1e-2, plate_d=30e-2):
    # plate_area = (p_result[1][2] - p_result[1][0])*(p_result[1][3] - p_result[1][1])

    _x, _y, w, h = yolov8_results['plate'].xywh[0].tolist()  # assuming one plate
    plate_d_pixels = (w + h) / 2
    meters_per_pixel = plate_d / plate_d_pixels

    foods_found = dict()
    # relative_sum_proteins, relative_sum_carbs, relative_sum_veg, relative_sum_calories_density = 0, 0, 0, 0
    for food_name, box in yolov8_results['food']:
        food_name = str(food_name).replace("_", " ")
        _x, _y, w, h = box.xywh[0].tolist()
        foods_found[food_name] = {
            "nutrients": {
                "servings": {},
            },
        }

        # food_area = (food[1][2] - food[1][0])*(food[1][3] - food[1][1])
        area = meters_per_pixel**2 * np.pi * w * h / 4
        volume = plate_h * area  # m^3
        density = DENSITY_MAP[food_name.replace(' ', '_')]  # kg/m^3
        weight = density * volume * 1000  # g
        foods_found[food_name]["measures"] = {
            "area": area,
            "volume": volume,
            "weight": weight,
        }

        # foods_found[food_name]["relative_area"] = food_area/plate_area

        # if food_name in "vegetables":
            # relative_sum_veg += 1*foods_found[food_name]["relative_area"]
            # continue

        normalizeScale = {
            "g": "g",
            "grm": "g",
            "mg": "mg",
            "kcal": "kcal",
        }

        food_data = __get_food_on_api(food_name)
        foods_found[food_name]["nutrients"]["servings"]["size"] = food_data["servingSize"]
        foods_found[food_name]["nutrients"]["servings"]["unit"] = normalizeScale[food_data["servingSizeUnit"].lower()]

        for n in food_data["foodNutrients"]:
            gotProtein = 'protein' in foods_found[food_name]["nutrients"]
            gotCarbs = 'carbs' in foods_found[food_name]["nutrients"]
            gotCalories = 'calories' in foods_found[food_name]["nutrients"]
            if gotProtein and gotCarbs and gotCalories:
                break
            if not gotProtein and 'protein' in str(n["nutrientName"]).lower():
                foods_found[food_name]["nutrients"]["protein"] = {
                    "unit": normalizeScale[n["unitName"].lower()],
                    "value": n["value"],
                }
            if not gotCarbs and 'carbohydrate' in str(n["nutrientName"]).lower():
                foods_found[food_name]["nutrients"]["carbs"] = {
                    "unit": normalizeScale[n["unitName"].lower()],
                    "value": n["value"],
                }
            if not gotCalories and 'energy' in str(n["nutrientName"]).lower():
                foods_found[food_name]["nutrients"]["calories"] = {
                    "unit": normalizeScale[n["unitName"].lower()],
                    "value": n["value"],
                }

        #TODO - Lidar com liquidos (unidade de volume)
        # if "l" in foods_found[food_name]["nutrients"]["servings"]["unit"].lower()

        for macro in ['protein', 'carbs', 'calories']:
            serving_size = foods_found[food_name]["nutrients"]["servings"]["size"]
            # ensure units is grams
            if foods_found[food_name]["nutrients"]["servings"]["unit"] == "mg":
                serving_size /= 1000
            if foods_found[food_name]["nutrients"]["servings"]["unit"] == "kg":
                serving_size *= 1000
            macro_size = foods_found[food_name]["nutrients"][macro]["value"]
            # ensure units is grams
            if foods_found[food_name]["nutrients"][macro]["unit"] == "mg":
                macro_size /= 1000
            if foods_found[food_name]["nutrients"][macro]["unit"] == "kg":
                macro_size *= 1000

            per_serving = macro_size / serving_size

            foods_found[food_name]["measures"][macro] = \
                foods_found[food_name]["measures"]["weight"] * per_serving

        if food_name in "vegetables":
            foods_found[food_name]["measures"]["vegetables"] = foods_found[food_name]["measures"]["weight"]
        else:
            foods_found[food_name]["measures"]["vegetables"] = 0

        # factor_protein, factor_carbs = 1, 1
        # if normalizeScale[str(foods_found[food_name]["nutrients"]["servings"]["unit"]).lower()] != normalizeScale[str(foods_found[food_name]["nutrients"]["protein"]["unit"]).lower()]:
        #     factor_protein = 0.001
        # foods_found[food_name]["nutrients"]["protein"]["percentage"] = factor_protein*(foods_found[food_name]["nutrients"]["protein"]["value"]/foods_found[food_name]["nutrients"]["servings"]["size"])*foods_found[food_name]["relative_area"]

        # if normalizeScale[str(foods_found[food_name]["nutrients"]["servings"]["unit"]).lower()] != normalizeScale[str(foods_found[food_name]["nutrients"]["carbs"]["unit"]).lower()]:
        #     factor_carbs = 0.001
        # foods_found[food_name]["nutrients"]["carbs"]["percentage"] = factor_carbs*(foods_found[food_name]["nutrients"]["carbs"]["value"]/foods_found[food_name]["nutrients"]["servings"]["size"])*foods_found[food_name]["relative_area"]

        # # VV Assumindo que a porção está sempre em gramas VV
        foods_found[food_name]["nutrients"]["calories"]["density"] = foods_found[food_name]["nutrients"]["calories"]["value"]/foods_found[food_name]["nutrients"]["servings"]["size"]

        # relative_sum_proteins += foods_found[food_name]["nutrients"]["protein"]["percentage"]
        # relative_sum_carbs += foods_found[food_name]["nutrients"]["carbs"]["percentage"]
        # relative_sum_calories_density += (foods_found[food_name]["nutrients"]["calories"]["density"] * foods_found[food_name]["relative_area"])

    totals = {
        macro: sum([foods_found[food_name]["measures"][macro] for food_name in foods_found.keys()])
        for macro in ['protein', 'carbs', 'calories', 'vegetables']
    }
    total_macros = sum([value for macro, value in totals.items() if macro != 'calories'])
    relatives = {
        macro: totals[macro] / total_macros
        for macro in ['protein', 'carbs', 'vegetables']
    }

    print("\nRealizando análise nutricional da refeição...\n")
    # if relative_sum_calories_density >= 4:
    #     print('ATENÇÃO: Densidade calórica relativa > 4 ==> Refeição potencialmente calórica!')

    # relative_sum_proteins = relative_sum_proteins if relative_sum_proteins <= 0.25 else 0.25
    # relative_sum_carbs = relative_sum_carbs if relative_sum_carbs <= 0.25 else 0.25
    # relative_sum_veg = relative_sum_veg if relative_sum_veg <= 0.5 else 0.5

    # average_health = relative_sum_proteins + relative_sum_carbs + relative_sum_veg
    # print("Média de salubidade: {:.2f} % saudável".format(average_health*100))
    # print("Proteínas: {:.2f} % da refeição".format(relative_sum_proteins*100))
    # print("Carboidratos: {:.2f} % da refeição".format(relative_sum_carbs*100))
    # print("Vegetais: {:.2f} % da refeição".format(relative_sum_veg*100))
    # print("Densidade calórica relativa e agregada: {:.2f} kcal/g".format(relative_sum_calories_density))

    ideals = {
        'protein': 0.25,
        'carbs': 0.25,
        'vegetables': 0.5,
    }
    average_health = 0
    for macro, ideal in ideals.items():
        average_health += (1/(ideal-1))*abs(relatives[macro] - ideal) + 1
    average_health /= 3

    print(f"Média de salubidade: {average_health*100:.2f} % saudável")
    print(f"Proteínas: {relatives['protein']*100:.2f} %(m/m) da refeição")
    print(f"Carboidratos: {relatives['carbs']*100:.2f} %(m/m) da refeição")
    print(f"Vegetais: {relatives['vegetables']*100:.2f} %(m/m) da refeição")

    return average_health*100, 0
