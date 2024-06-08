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

        window_name = "Realtime"
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.resizeWindow(window_name, 800, 600)
        cv.imshow(window_name, frame)
        # move_window_to_center("Realtime", *frame.shape[:2][::-1])

        if cv.waitKey(0) == ord("q"):
            break
    cv.destroyAllWindows()

if __name__ == "__main__":
    app()
