import json
import cv2 as cv
import bbox_visualizer as bbv # visualizador de bounding boxes
import numpy as np
import os
import shutil

def yolo_to_coordinates(x, y, w, h):
    """
    Converte caixas no formato YOLO para o formato Pixels. (zidane.jpg) (x1, y1, x2, y2)
    Precisamos do tamanho da imagem original em pixels para a conversão (1280x720)

    Entrada: x, y, w, h (em %) - formato YOLO
    """
    print(x, y, w, h)
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return [round(x1 * 1280), round(y1 * 720), round(x2 * 1280), round(y2 * 720)]

def yolo_conversion_test():
    """
    Testei converter as caixas no formato YOLO para o formato Pixels, e mostrei
    a imagem final para ver se deu certo a conversão. (zidane.jpg)
    """
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    # Image with YOLO annotations
    yolo_img = cv.imread(os.path.join(curr_dir, "zidane.jpg"))
    yolo_labels = ["person", "person", "tie"]
    yolo_bboxes = [[0.481719, 0.634028, 0.690625, 0.713278], [0.741094, 0.524306, 0.314750, 0.933389], [0.364844, 0.795833, 0.078125, 0.400000]]
    # Convert YOLO annotations to normal bounding boxes
    normal_bboxes = [yolo_to_coordinates(*bbox) for bbox in yolo_bboxes]
    print(normal_bboxes)
    # Draw bounding boxes and labels
    img_with_boxes = bbv.draw_multiple_rectangles(yolo_img, normal_bboxes)
    img_with_boxes = bbv.add_multiple_labels(img_with_boxes, yolo_labels, normal_bboxes)
    cv.imshow("image", img_with_boxes)
    cv.waitKey(0)

def coordinates_to_yolo(width, height, x_min, y_min, x_max, y_max):
    """
    Converte caixas no formato Pixels para o formato YOLO.
    Recebe o tamanho da imagem original em pixels (width, height) e as coordenadas da caixa em pixels,
    (x_min, y_min, x_max, y_max)
    """
    # x_min, y_min, x_max, y_max (in px) to x, y, w, h (in %)
    dw = 1./width
    dh = 1./height
    x = (x_min + x_max)/2.0
    y = (y_min + y_max)/2.0
    w = x_max - x_min
    h = y_max - y_min
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def bbox_from_seglist(seg_list):
    """
    Cria uma bounding box a partir de uma lista de segmentos.
    """
    seg_points = [(seg_list[i], seg_list[i+1]) for i in range(0, len(seg_list), 2)]
    x_min = min([p[0] for p in seg_points])
    y_min = min([p[1] for p in seg_points])
    x_max = max([p[0] for p in seg_points])
    y_max = max([p[1] for p in seg_points])
    return [round(x_min), round(y_min), round(x_max), round(y_max)]

def create_yolo_dataset_annotations():
    """
    Cria um diretório com as anotações no formato YOLO e um arquivo de texto com as classes encontradas.
    """
    # Diretório atual
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    # Localização das anotações do dataset
    # No meu caso, estavam numa pasta chamada "test" e em um arquivo chamado "annotations.json"
    dataset_json = json.load(open(os.path.join(curr_dir, "test", "annotations.json")))

    # Diretório com as anotações no formato YOLO e com as classes encontradas
    annotations_directory = os.path.join(curr_dir, "test", "yoloV5_annotations")
    if (os.path.exists(annotations_directory)):
        print("Diretório de mesmo nome já existe. Deseja removê-lo? (y/n)")
        if input() == "y":
            shutil.rmtree(annotations_directory)
        else:
            print("Abortando a criação do diretório de anotações YOLO...")
            return
    print("Criando e preenchendo diretório de anotações YOLO...")
    os.makedirs(annotations_directory, exist_ok=True)

    categories = {}
    for annotation in dataset_json["annotations"]:
        img_filename = str(annotation["image_id"]).zfill(6) + ".txt"
        # size
        img_id = annotation["image_id"]
        img_width, img_height = 0, 0
        for image in dataset_json["images"]:
            if image["id"] == img_id:
                img_width = image["width"]
                img_height = image["height"]
                break
        # category (class)
        category_id = annotation["category_id"]
        category_name = ""
        for category in dataset_json["categories"]:
            if category["id"] == category_id:
                category_name = category["name"]
                break
        if category_name not in categories:
                categories[category_name] = len(categories)
        category_yolo_id = categories[category_name]
        # bboxes and append to file
        for segmentation in annotation["segmentation"]:
            bbox = bbox_from_seglist(segmentation)
            yolo_bbox = coordinates_to_yolo(img_width, img_height, *bbox)
            with open(os.path.join(annotations_directory, img_filename), "a") as f:
                f.write("{} {} {} {} {}\n".format(category_yolo_id, *yolo_bbox))

    # Exportar as classes para um arquivo de texto
    with open(os.path.join(annotations_directory, "_classes.txt"), "w") as f:
        for category, yolo_id in sorted(categories.items(), key=lambda x: x[1]):
            f.write("{}: {}\n".format(yolo_id, category))

    print("Anotações YOLO criadas com sucesso!")

def visualize_segmentation(img, segmentations, category_name):
    """
    Visualiza a segmentação de uma imagem, mostrando a máscara da segmentação e o nome da classe.
    """
    # Create mask from segmentation
    mask = np.zeros_like(img)
    for s in range(len(segmentations)):
        # Create mask from segmentation
        seg = list(map(int, segmentations[s]))
        seg_points = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
        cv.polylines(mask, [np.array(seg_points)], isClosed=True, color=(25,25,255), thickness=3)
        cv.fillPoly(mask, [np.array(seg_points)], color=(159,156,1))
        result = cv.addWeighted(img, 0.5, mask, 0.5, 0)
    # Show segmentation image
    cv.putText(result, category_name, (seg_points[0][0], seg_points[0][1]), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,222), 2, 1)
    cv.imshow("image", result)
    k = cv.waitKey(0)
    return k

def visualize_bounding_boxes(img, bboxes, categories):
    """
    Visualiza as bounding boxes de uma imagem, mostrando as caixas e os nomes das classes.
    """
    img_with_boxes = bbv.draw_multiple_rectangles(img, bboxes)
    img_with_boxes = bbv.add_multiple_labels(img_with_boxes, categories, bboxes)
    cv.imshow("image", img_with_boxes)
    k = cv.waitKey(0)
    return k

def visualize_data_set():
    """
    Visualiza o dataset de teste, mostrando as segmentações e bounding boxes de cada imagem.
    As anotações utilizadas estão no arquivo "annotations.json" no atributo "annotations".
    Para obter os nomes das classes, é necessário acessar o atributo "categories" do mesmo arquivo.

    PARA PARAR DE VISUALIZAR AS IMAGENS, PRESSIONE A TECLA "ESC" (27).
    """

    # Localização das anotações do dataset
    # No meu caso, estavam numa pasta chamada "test" e em um arquivo chamado "annotations.json"
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_json = json.load(open(os.path.join(curr_dir, "test", "annotations.json")))

    for annotation in dataset_json["annotations"]:
        # Get image_id from the annotation and then get the image jpg
        imagem = str(annotation["image_id"])
        # Localização das imagens (no meu caso, na pasta "test/images")
        img = cv.imread(os.path.join(curr_dir, "test", "images", "{}.jpg".format(imagem.zfill(6))) ) # load image

        # Get category_id from the annotation and then the name
        category_id = annotation["category_id"]
        category_name = ""
        for category in dataset_json["categories"]:
            if category["id"] == category_id:
                category_name = category["name"]
                break

        # Get segmentations list from the annotation
        segmentations = annotation["segmentation"]
        # Visualizar segmentação
        if visualize_segmentation(img, segmentations, category_name) == 27:
            break

        # # Create bounding box from segmentation
        seg_bboxes = [bbox_from_seglist(seg) for seg in segmentations]
        # Visualizar segmentação
        if visualize_bounding_boxes(img, seg_bboxes, [category_name]*len(segmentations)) == 27:
            break

        """
        As linhas abaixo tentam usar as caixas originais do dataset.
        Pesquisei que o formato original delas talvez seja [y, x, height, width], e converti elas para [x_min, y_min, x_max, y_max]
        """
        # Get the *original* bounding box from the annotation (*original* format: [y, x, height, width])
        # and then convert it to [x_min, y_min, x_max, y_max]
        # bbox = annotation["bbox"]
        # bbox[0], bbox[1] = bbox[1], bbox[0] # swap x and y
        # bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3] # convert to xmax, ymax
        # # Show *original* bounding box
        # bbox = list(map(int, bbox)) # convert to int
        # img_with_box = bbv.draw_rectangle(img, bbox, bbox_color=(0,0,0))
        # img_label = bbv.add_label(img_with_box, category_name, bbox)
        # cv.imshow("image", img_label)
        # k = cv.waitKey(0)
        # if k == 27:
        #     break

if __name__ == "__main__":
    visualize_data_set()
    # create_yolo_dataset_annotations() # Criar anotações no formato YOLO e exportar as classes
