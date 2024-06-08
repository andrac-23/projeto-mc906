import os
import json
import shutil
import sys
import yaml

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

def create_yolo_dataset_files_and_annotations(version, letter):
    """
    Cria um diretório com as anotações no formato YOLO e um arquivo de texto com as classes encontradas.
    """
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    dataset_json = json.load(open(os.path.join(curr_dir, "v{}".format(version), "v{}_{}".format(version, letter), "v{}_{}_annotations.json".format(version, letter))))

    # Diretório para receber as anotações no formato YOLOv8
    annotations_directory = os.path.join(curr_dir, "v{}".format(version), "v{}_{}".format(version, letter), "v{}_{}_yoloV8_annotations".format(version, letter))
    if (os.path.exists(annotations_directory)):
        print("Diretório de mesmo nome já existe. Deseja removê-lo? (y/n)")
        if input() == "y":
            shutil.rmtree(annotations_directory)
        else:
            print("Abortando a criação do diretório de anotações YOLO...")
            return
    print("Criando e preenchendo o diretório com as anotações YOLO...")
    os.makedirs(annotations_directory, exist_ok=True)

    # Salvar tamanhos das imagens
    img_sizes = {}
    for image in dataset_json["images"]:
        img_sizes[image["id"]] = (image["width"], image["height"])

    # Salvar nomes das categorias
    category_names = {}
    for category in dataset_json["categories"]:
        category_names[category["id"]] = category["name"]

    # Criação das anotações YOLO
    categories, curr_annotation = {}, 0
    category_instance_count = {}
    for annotation in dataset_json["annotations"]:
        img_filename = str(annotation["image_id"]).zfill(6) + ".txt"
        img_id = annotation["image_id"]
        # Obter img width and height
        img_width, img_height = img_sizes[img_id]
        # Obter o nome da categoria a partir do id
        category_id = annotation["category_id"]
        category_name = category_names[category_id]
        # Criar o ID da categoria para o yolov8
        if category_name not in categories:
            categories[category_name] = len(categories)
        category_yolo_id = categories[category_name]

        # Obter a contagem de instâncias da categoria
        if category_yolo_id not in category_instance_count:
            category_instance_count[category_yolo_id] = 0
        category_instance_count[category_yolo_id] += len(annotation["segmentation"])

        # Obter a bounding box a partir da segmentação (caixas vieram erradas/diferentes no annotations original) e escrever na label
        for segmentation in annotation["segmentation"]:
            bbox = bbox_from_seglist(segmentation)
            yolo_bbox = coordinates_to_yolo(img_width, img_height, *bbox)
            with open(os.path.join(annotations_directory, img_filename), "a") as f:
                f.write("{} {} {} {} {}\n".format(category_yolo_id, *yolo_bbox))
        # Imprimir progresso
        curr_annotation += 1
        sys.stdout.write("\rProgresso atual: {:.2f}%".format(100*curr_annotation/len(dataset_json["annotations"])))

    # Exportar class instance count
    with open(os.path.join(curr_dir, "v{}".format(version), "v{}_{}".format(version, letter), "v{}_{}_class_instance_count.txt".format(version, letter)), "w") as f:
        for category_yolo_id, count in category_instance_count.items():
            f.write("{}: {}\n".format(category_yolo_id, count))

    print("\nAnotações YOLO criadas com sucesso!")

    print("Criando diretórios de arquivos YOLO e arquivo data.yaml...")
    # Criar diretórios de arquivos yolo
    yolo_files_dir = os.path.join(curr_dir, "v{}".format(version), "v{}_{}".format(version, letter), "v{}_{}_yoloV8_files".format(version, letter))
    os.makedirs(os.path.join(yolo_files_dir, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(yolo_files_dir, "train", "labels"), exist_ok=True)
    os.makedirs(os.path.join(yolo_files_dir, "val", "images"), exist_ok=True)
    os.makedirs(os.path.join(yolo_files_dir, "val", "labels"), exist_ok=True)
    os.makedirs(os.path.join(yolo_files_dir, "test", "images"), exist_ok=True)
    os.makedirs(os.path.join(yolo_files_dir, "test", "labels"), exist_ok=True)

    # Criar o data.yaml
    with open(os.path.join(yolo_files_dir, "data.yaml"), "w") as f:
        yaml.safe_dump({
            #"path": os.path.relpath(yolo_files_dir, curr_dir),
            "train": "train",
            "val": "val",
            "test": "test",
            "nc": len(categories),
            "names": list(categories.keys())
        }, f)

    print("Diretórios de arquivos YOLO e arquivo data.yaml criados com sucesso!\nFim do script")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso incorreto, esperado: python3 create_yolo_labels.py <versao> <letra>")
        sys.exit(1)
    versao, letra = int(sys.argv[1]), sys.argv[2].upper()
    if letra not in ["A", "M"] or versao not in [1, 2, 3]:
        print("Argumentos inválidos")
        sys.exit(1)
    create_yolo_dataset_files_and_annotations(versao, letra)
