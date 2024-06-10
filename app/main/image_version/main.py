import cv2 as cv
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from modules import modules


def app():
    root_dir = os.path.abspath(modules.build_path(os.path.dirname(__file__), '..', '..', '..'))
    images_dir = modules.build_path(root_dir, 'app', 'main', 'image_version', 'images')
    ia_models_dir = modules.build_path(root_dir, 'ia_models', 'final')

    for filename in os.listdir(images_dir):
        image_path = modules.build_path(images_dir, filename)
        frame = cv.imread(image_path)

        yolo_results = modules.get_yolo_detection_results(frame, ia_models_dir)
        frame = modules.insert_food_regions_detected(frame, yolo_results)

        window_app = "App - Image Version"
        cv.namedWindow(window_app, cv.WINDOW_NORMAL)
        cv.resizeWindow(window_app, 800, 600)
        cv.imshow(window_app, frame)

        modules.show_metrics_analisys(yolo_results)
        # window_metrics = "Metrics"
        # cv.namedWindow(window_metrics, cv.WINDOW_NORMAL)
        # cv.resizeWindow(window_metrics, 800, 600)
        # cv.imshow(window_metrics, metrics_frame)
        # move_window_to_center("Realtime", *frame.shape[:2][::-1])

        print("\nPressione a tecla \"space bar\" na janela da imagem verificada, para continuar a analise das proximas imagens.\nCaso desejar encerrar antes, pressione 'q'.")
        key = cv.waitKey()
        if key == ord("q"):
            break

    cv.destroyAllWindows()

if __name__ == "__main__":
    app()
