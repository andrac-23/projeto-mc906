import cv2 as cv
import numpy as np
import tkinter as tk
import bbox_visualizer as bbv
# from ultralytics import YOLO

def insert_title_to_frame(frame, text):
    font_scale = 1
    font_thickness = 2
    (text_width, text_height), baseline = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    height, width = frame.shape[:2]
    x = (width - text_width) // 2 # centralizado
    y = int(0.075 * height) # 7.5% da altura
    cv.putText(frame, text, (x, y), cv.FONT_HERSHEY_COMPLEX, font_scale, (255,255,255), font_thickness)
    return

def insert_rectangle_to_frame(frame):
    # forma do retangulo
    frame_height, frame_width = frame.shape[:2]
    top_left_point = (int(0.87 * frame_width), int(0.075 * frame_height))
    bottom_right_point = (int(0.97 * frame_width), int(0.925 * frame_height))

    rect_width = bottom_right_point[0] - top_left_point[0]
    rect_height = bottom_right_point[1] - top_left_point[1]

    # preencher retangulo
    fill_percentage = 0.38  # 60% fill
    fill_height = int(rect_height * fill_percentage)
    # Define the filled rectangle parameters starting from the base
    fill_start_point = (top_left_point[0], bottom_right_point[1] - fill_height)
    fill_end_point = bottom_right_point
    cv.rectangle(frame, fill_start_point, fill_end_point, (52, 57, 244), cv.FILLED)

    # texto na base do retangulo
    text = "38%"
    font_scale = 1
    font_thickness = 1
    (text_width, text_height), baseline = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    x = top_left_point[0] + (rect_width - text_width) // 2
    y = bottom_right_point[1] - text_height - 50
    cv.putText(frame, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    cv.rectangle(frame, top_left_point, bottom_right_point, (8, 184, 27), 2)
    return

def adicionar_interface_inicial(frame):
    insert_title_to_frame(frame, "PROJETO MC906")
    insert_rectangle_to_frame(frame)

def get_screen_resolution():
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    return screen_width, screen_height

def move_window_to_center(window_name, width, height):
    screen_width, screen_height = get_screen_resolution()
    screen_height = int(screen_height * 0.90) # 90% da tela
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    cv.moveWindow(window_name, x, y)
    return

def get_yolo_detection_results(frame):
    # food_model = YOLO(r"localizacao_do_modelo.pt")
    # f_results = food_model.predict([frame])[0]
    # f_cls_and_bboxes = [(f_results.names[cls], list(map(int, bbox))) for cls, bbox in zip(f_results.boxes.cls, f_results.boxes.xyxy)]

    # plate_model = YOLO(r"localizacao_do_modelo.pt")
    # p_results = plate_model.predict([frame])[0]
    # p_cls_and_bbox = ["prato", list(map(int, p_results.boxes.xyxy[0]))] # só tem um prato por imagem

    # return [p_cls_and_bbox, f_cls_and_bboxes]
    return [["prato", [386, 126, 872, 588]], [("bread", [416, 136, 802, 548])]] # exemplo de retorno

def analisar_saude_do_prato(p_result, f_results):
    # @TODO
    pass

def adicionar_resultados_yolo(frame, yolov8_results, add_plate=True):
    p_result, f_results = yolov8_results[0], yolov8_results[1]

    # resultados = analisar_saude_do_prato(p_result, f_results)

    bboxes = [p_result[1]] + [bbox for _, bbox in f_results]
    labels = [p_result[0]] + [cls for cls, _ in f_results]

    if add_plate:
        frame = bbv.draw_rectangle(frame, p_result[1])
        frame = bbv.add_label(frame, "prato", p_result[1])

    frame = bbv.draw_multiple_rectangles(frame, bboxes[1:], bbox_color=(0, 0, 189), is_opaque=True, alpha=0.3)
    frame = bbv.add_multiple_labels(frame, labels[1:], bboxes[1:], text_bg_color=(193, 193, 163), top=False)

    return frame

def app():
    # cap = cv.VideoCapture(0)
    # get current webcam resolution
    # cap.set(cv.CAP_PROP_FRAME_WIDTH, 1600)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 900)
    # if not cap.isOpened():
    #     print("Erro: Não foi possível abrir a câmera")
    #     return
    while True:
        # ret, frame = cap.read()
        # if not ret:
        #     print("Erro: Não foi possível capturar o frame")
        #     continue

        # frame = cv.imread(r"localizacao.jpg")
        frame = cv.imread(r"localizacao.jpeg")

        yolo_results = get_yolo_detection_results(frame)

        adicionar_interface_inicial(frame)
        frame = adicionar_resultados_yolo(frame, yolo_results)

        cv.imshow("Realtime", frame)
        #move_window_to_center("Realtime", *frame.shape[:2][::-1])

        # TODO escolher o tamanho da janela
        # TODO deixar a janela no centro da tela
        # cv.resizeWindow("Realtime")
        if cv.waitKey(0) == ord("q"):
            break
    # cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    app()
