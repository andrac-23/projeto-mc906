import cv2 as cv
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from modules import modules

def app():
    root_dir = os.path.abspath(modules.build_path(os.path.dirname(__file__), '..', '..', '..'))
    ia_models_dir = modules.build_path(root_dir, 'ia_models', 'final')
    
    cap = cv.VideoCapture(0)
    # get current webcam resolution
    # cap.set(cv.CAP_PROP_FRAME_WIDTH, 1600)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 900)
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro: Não foi possível capturar o frame")
            continue

        yolo_results = modules.get_yolo_detection_results(frame, ia_models_dir)
        frame = modules.insert_food_regions_detected(frame, yolo_results)
        
        window_name = "App - Realtime Version"
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.resizeWindow(window_name, 800, 600)
        cv.imshow(window_name, frame)
        # move_window_to_center("Realtime", *frame.shape[:2][::-1])

        modules.show_metrics_analisys(yolo_results)
        # TODO escolher o tamanho da janela
        # TODO deixar a janela no centro da tela
        # cv.resizeWindow("Realtime") #type: ignore
        if cv.waitKey(1) == ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    app()
