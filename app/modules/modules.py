from ultralytics import YOLO
from ultralytics.engine.results import Boxes
import cv2 as cv
import tkinter as tk
import requests
from dotenv import load_dotenv
import os
import numpy as np
import matplotlib.pyplot as plt

food_cache = dict()

def get_yolo_detection_results(frame, models_dir):
    print("Realizando detecção de objetos na imagem atual...")
    food_model = YOLO(build_path(models_dir,'food.pt'))
    f_results = food_model.predict([frame])[0]
    if not isinstance(f_results.boxes, Boxes):
        raise RuntimeError("no plate boxes returned")
    f_cls_and_bboxes = [(f_results.names[int(cls)], list(map(int, bbox))) for cls, bbox in zip(f_results.boxes.cls, f_results.boxes.xyxy)]

    plate_model = YOLO(build_path(models_dir,'plate.pt'))
    p_results = plate_model.predict([frame])[0]
    if not isinstance(p_results.boxes, Boxes):
        raise RuntimeError("no food boxes returned")
    p_cls_and_bbox = ["prato", list(map(int, p_results.boxes.xyxy[0]))]

    return [p_cls_and_bbox, f_cls_and_bboxes]

def draw_bbox_and_label(frame, name, bbox, bbox_color=(0, 255, 0), text_color=(0, 0, 0), paint_inside=False, paint_alfa=0.15, paint_color=(0, 255, 0)):
    """ Credits to: https://github.com/shoumikchow/bbox-visualizer/ """

    # Drawing the bounding box
    thickness = 2
    x_min, y_min, x_max, y_max = bbox
    cv.rectangle(frame, (x_min, y_min), (x_max, y_max), bbox_color, thickness)
    if paint_inside:
        copy = frame.copy()
        cv.rectangle(copy, (x_min, y_min), (x_max, y_max), paint_color, cv.FILLED)
        cv.addWeighted(copy, paint_alfa, frame, 1-paint_alfa, 0, frame)

    # Drawing the label
    font = cv.FONT_HERSHEY_SIMPLEX
    thickness = 1
    size = 0.86 # @TODO: make it dynamic based on the size of the bounding box    
    background_color = bbox_color
    (text_width, text_height), baseline = cv.getTextSize(name, font, size, thickness)
    # Background of the label
    rec_coordinates = [int(0.999*x_min), int(1*y_min), x_min + text_width, y_min - text_height - int((10 * size))]
    cv.rectangle(frame, (rec_coordinates[0], rec_coordinates[1]), (rec_coordinates[2] + 5, rec_coordinates[3]), background_color, cv.FILLED)
    # Put the text on top of the bounding box
    cv.putText(frame, name, (x_min, y_min - int(5.5 * size)), font, size, text_color, thickness, cv.LINE_AA)
    cv.putText(frame, name, (x_min, y_min - 1 - int(5.5 * size)), font, size, text_color, thickness, cv.LINE_AA) # workaround for custom thickness

def insert_food_regions_detected(frame, yolov8_results):
    p_result, f_results = yolov8_results[0], yolov8_results[1]

    # Desenhar alimentos
    for item in f_results:
        name, bounding_box = item[0], item[1]
        draw_bbox_and_label(frame, name, bounding_box)

    # Desenhar prato
    name, bounding_box = p_result[0], p_result[1]
    draw_bbox_and_label(frame, name, bounding_box, bbox_color=(0, 0, 0), text_color=(255, 255, 255), paint_inside=True, paint_alfa=0.15, paint_color=(0, 0, 0))

    return frame

def build_path(*paths):
    return os.path.join(*paths)

def show_metrics_analisys(yolov8_results):
    p_result, f_results = yolov8_results[0], yolov8_results[1]
    resp = __healthy_analisys(p_result, f_results)

    percentages = resp.values()
    labels = resp.keys()

    __plot_metrics(percentages, labels)
    # return __draw_metrics(percentages, labels)
    
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

    if food_name in food_cache:
        # print(f"Cache hit: {food_name}")
        return food_cache[food_name]    

    response = requests.get(api_url, params=params)    

    if response.status_code == 200:
        food_cache[food_name] = response.json()["foods"][0]
        return response.json()["foods"][0]
    else:
        raise Exception(f"Erro na requisição: {response.status_code}")

def __healthy_analisys(p_result, f_results) -> dict:
    plate_area = (p_result[1][2] - p_result[1][0])*(p_result[1][3] - p_result[1][1])

    foods_found = dict()
    relative_sum_proteins, relative_sum_carbs, relative_sum_veg, relative_sum_calories_density = 0, 0, 0, 0
    for food in f_results:
        food_name = str(food[0]).replace("_", " ")
        foods_found[food_name] = {
            "nutrients": {
                "servings": {}
            }
        }

        food_area = (food[1][2] - food[1][0])*(food[1][3] - food[1][1])

        foods_found[food_name]["relative_area"] = food_area/plate_area

        if food_name in "vegetables":
            relative_sum_veg += 1*foods_found[food_name]["relative_area"]
            continue

        food_data = __get_food_on_api(food_name)
        foods_found[food_name]["nutrients"]["servings"]["size"] = food_data["servingSize"]
        foods_found[food_name]["nutrients"]["servings"]["unit"] = food_data["servingSizeUnit"]

        for n in food_data["foodNutrients"]:
            gotProtein = 'protein' in foods_found[food_name]["nutrients"]
            gotCarbs = 'carbs' in foods_found[food_name]["nutrients"]
            gotCalories = 'calories' in foods_found[food_name]["nutrients"]
            if gotProtein and gotCarbs and gotCalories:
                break
            if not gotProtein and 'protein' in str(n["nutrientName"]).lower():
                foods_found[food_name]["nutrients"]["protein"] = {}
                foods_found[food_name]["nutrients"]["protein"]["unit"] = n["unitName"]
                foods_found[food_name]["nutrients"]["protein"]["value"] = n["value"]
            if not gotCarbs and 'carbohydrate' in str(n["nutrientName"]).lower():
                foods_found[food_name]["nutrients"]["carbs"] = {}
                foods_found[food_name]["nutrients"]["carbs"]["unit"] = n["unitName"]
                foods_found[food_name]["nutrients"]["carbs"]["value"] = n["value"]
            if not gotCalories and 'energy' in str(n["nutrientName"]).lower():
                foods_found[food_name]["nutrients"]["calories"] = {}
                foods_found[food_name]["nutrients"]["calories"]["unit"] = n["unitName"]
                foods_found[food_name]["nutrients"]["calories"]["value"] = n["value"]

        #TODO - Lidar com liquidos (unidade de volume)
        # if "l" in foods_found[food_name]["nutrients"]["servings"]["unit"].lower()

        normalizeScale = {
            "g": "g",
            "grm": "g",
            "mg": "mg"
        }

        factor_protein, factor_carbs = 1, 1
        if normalizeScale[str(foods_found[food_name]["nutrients"]["servings"]["unit"]).lower()] != normalizeScale[str(foods_found[food_name]["nutrients"]["protein"]["unit"]).lower()]:
            factor_protein = 0.001
        foods_found[food_name]["nutrients"]["protein"]["percentage"] = factor_protein*(foods_found[food_name]["nutrients"]["protein"]["value"]/foods_found[food_name]["nutrients"]["servings"]["size"])*foods_found[food_name]["relative_area"]

        if normalizeScale[str(foods_found[food_name]["nutrients"]["servings"]["unit"]).lower()] != normalizeScale[str(foods_found[food_name]["nutrients"]["carbs"]["unit"]).lower()]:
            factor_carbs = 0.001
        foods_found[food_name]["nutrients"]["carbs"]["percentage"] = factor_carbs*(foods_found[food_name]["nutrients"]["carbs"]["value"]/foods_found[food_name]["nutrients"]["servings"]["size"])*foods_found[food_name]["relative_area"]

        # VV Assumindo que a porção está sempre em gramas VV
        foods_found[food_name]["nutrients"]["calories"]["density"] = foods_found[food_name]["nutrients"]["calories"]["value"]/foods_found[food_name]["nutrients"]["servings"]["size"]

        relative_sum_proteins += foods_found[food_name]["nutrients"]["protein"]["percentage"]
        relative_sum_carbs += foods_found[food_name]["nutrients"]["carbs"]["percentage"]
        relative_sum_calories_density += (foods_found[food_name]["nutrients"]["calories"]["density"] * foods_found[food_name]["relative_area"])

    print("\nRealizando análise nutricional da refeição...\n")
    if relative_sum_calories_density >= 4:
        print('ATENÇÃO: Densidade calórica relativa > 4 ==> Refeição potencialmente calórica!')

    # relative_sum_proteins = relative_sum_proteins if relative_sum_proteins <= 0.25 else 0.25
    # relative_sum_carbs = relative_sum_carbs if relative_sum_carbs <= 0.25 else 0.25
    # relative_sum_veg = relative_sum_veg if relative_sum_veg <= 0.5 else 0.5

    average_health = (relative_sum_proteins/0.25 + relative_sum_carbs/0.25 + relative_sum_veg/0.5)/3
    print("Score: {:.2f} % saudável".format(average_health*100))
    print("Proteínas: {:.2f} % da refeição".format(relative_sum_proteins*100))
    print("Carboidratos: {:.2f} % da refeição".format(relative_sum_carbs*100))
    print("Vegetais: {:.2f} % da refeição".format(relative_sum_veg*100))
    print("Densidade calórica relativa e agregada: {:.2f} kcal/g".format(relative_sum_calories_density))

    dict_resp = {
        'score': round(average_health, 4),
        'proteinas': round(relative_sum_proteins, 4),
        'carboidratos': round(relative_sum_carbs, 4),
        'vegetais': round(relative_sum_veg, 4),
        'kcal/g_percentual': round(relative_sum_calories_density/4, 4) # até 4 é adequado (100%); mais q 4, é perigoso
    }
    return dict_resp

def __plot_metrics(percentages, labels):
    # Porcentagens de cada métrica
    percentages_scaled = [p * 100 for p in percentages]

    # Criar figura com tamanho específico
    plt.figure(figsize=(12, 6))

    # Criar barras horizontais
    bars = plt.barh(labels, percentages_scaled, color='skyblue')

    # Adicionar valores nas barras
    for bar, percent in zip(bars, percentages_scaled):
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{percent}%', 
                va='center', ha='left', color='black', fontsize=10)

    # Inverter a ordem das métricas
    plt.gca().invert_yaxis()

    # Adicionar título e rótulos
    plt.xlabel('Porcentagem')
    plt.title('Métricas')

    # Exibir o gráfico
    plt.show()

def __draw_metrics(percentages, labels):
    # Tamanho da janela
    width, height = 400, 300

    # Criar uma imagem branca com o tamanho desejado
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Margens para os retângulos e rótulos
    margin_x, margin_y = 20, 20
    label_height = 20
    label_margin = 3  # Espaçamento entre o rótulo e a barra

    # Número de métricas
    num_metrics = len(percentages)

    # Tamanho dos retângulos
    inner_width = width - 2 * margin_x
    bar_height = (height - 2 * margin_y - (num_metrics - 1) * label_height) // num_metrics

    # Desenhar as barras e adicionar rótulos
    for i, (percent, label) in enumerate(zip(percentages, labels)):
        # Calcular o comprimento da barra com base na porcentagem
        bar_length = int(inner_width * percent)
        
        # Calcular a posição vertical da barra
        y = margin_y + i * (bar_height + label_height)
        
        # Desenhar o retângulo interno (preenchimento)
        cv.rectangle(image, (margin_x, y), (margin_x + bar_length, y + bar_height), (255, 0, 0), -1)  # Azul

        # Desenhar o retângulo externo
        cv.rectangle(image, (margin_x, y), (width - margin_x, y + bar_height), (0, 0, 0), 2)
        
        # Adicionar rótulo acima da barra
        cv.putText(image, label, (margin_x, y - label_margin), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image

def __insert_vertical_bar_healthy_score(frame, yolov8_results):
    #forma do retangulo
    frame_height, frame_width = frame.shape[:2]
    top_left_point = (int(0.87 * frame_width), int(0.075 * frame_height))
    bottom_right_point = (int(0.97 * frame_width), int(0.925 * frame_height))

    rect_width = bottom_right_point[0] - top_left_point[0]
    rect_height = bottom_right_point[1] - top_left_point[1]

    p_result, f_results = yolov8_results[0], yolov8_results[1]
    resp = __healthy_analisys(p_result, f_results)

    # preencher retangulo
    fill_percentage = resp["score"]
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