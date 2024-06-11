import os
from argparse import ArgumentParser
from pathlib import Path

import cv2 as cv

from app.lib import modules


def image_aux(models: Path | str, image_path: Path | str,) -> None:
    frame = cv.imread(image_path)
    yolo_results = modules.get_yolo_detection_results(frame, models)

    _, width, _ = frame.shape
    top, bottom, left, right = 0, 0, 0, int(.2 * width)
    # print(f'{(height, width)=}')
    padded_image = cv.copyMakeBorder(frame, top, bottom, left, right, cv.BORDER_CONSTANT, value=[255, 255, 255])
    frame = modules.insert_food_regions_detected(padded_image, yolo_results)

    window_name = "App - Image Version"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    height, width, _ = frame.shape
    cv.resizeWindow(window_name, width, height)
    cv.imshow(window_name, frame)
    # move_window_to_center("Realtime", *frame.shape[:2][::-1])

    print("\nPressione 'q' para encerrar. Pressione qualquer outra tecla para continuar.")
    key = cv.waitKey()
    while key != ord("q"):
        pass

def image(models: Path | str, path: Path | str):
    if os.path.isdir(path):
        for filename in os.listdir(path):
            image_path = modules.build_path(path, filename)
            image_aux(models, image_path)
    elif os.path.isfile(path):
        image_aux(models, path)
    cv.destroyAllWindows()

def realtime(models: Path | str):
    cap = cv.VideoCapture(0)
    # get current webcam resolution
    # cap.set(cv.CAP_PROP_FRAME_WIDTH, 1600)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 900)
    try:
        while not cap.isOpened():
            pass
    except Exception:
        print("Erro: Não foi possível abrir a câmera")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro: Não foi possível capturar o frame")
            continue

        yolo_results = modules.get_yolo_detection_results(frame, models)
        frame = modules.insert_food_regions_detected(frame, yolo_results)

        window_name = "Realtime"
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.resizeWindow(window_name, 800, 600)
        cv.imshow(window_name, frame)
        # move_window_to_center("Realtime", *frame.shape[:2][::-1])

        # TODO escolher o tamanho da janela
        # TODO deixar a janela no centro da tela
        # cv.resizeWindow("Realtime") #type: ignore
        if cv.waitKey(0) == ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()


def main() -> int:
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help', required=True)

    parser_image = subparsers.add_parser('image', help='image help')
    parser_image.add_argument('-p', '--path',
                              required=True)
    parser_image.add_argument('mode', const='image', action='store_const')
    parser_image.add_argument('-m', '--models', required=True, help='models directory')

    parser_realtime = subparsers.add_parser('realtime', help='image help')
    parser_realtime.add_argument('mode', const='realtime', action='store_const')
    parser_realtime.add_argument('-m', '--models', required=True, help='models directory')

    args = parser.parse_args()
    match args.mode:
        case 'image':
            image(args.models, args.path)
        case 'realtime':
            realtime(args.models)

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
