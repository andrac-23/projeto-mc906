import os
import json
import glob

def get_class_id_from_dataset(dataset_json, class_name):
    for category in dataset_json["categories"]:
        if category["name"] == class_name:
            return category["id"]
    return None

def get_useless_images(old_dataset_json, classes_removidas):
    total_images, used_images = set(), set()

    for image in old_dataset_json["images"]:
        total_images.add(image["id"])

    for annotation in old_dataset_json["annotations"]:
        if annotation["category_id"] in classes_removidas:
            continue
        used_images.add(annotation["image_id"])

    return total_images - used_images

def create_new_ds_version(old_version, old_letter, new_version, new_letter):
    print("Criando dataset v{}_{} a partir do dataset v{}_{}".format(new_version, new_letter, old_version, old_letter))

    curr_location = os.path.dirname(os.path.abspath(__file__))

    old_dataset_location = os.path.join(curr_location, "v{}".format(old_version), "v{}_{}".format(old_version, old_letter), "v{}_{}_annotations.json".format(old_version, old_letter))
    old_dataset_json = json.load(open(old_dataset_location))

    new_class_id = {} # new_class => new_id
    old_id_to_new_id = {} # id das classes antigas => id das novas classes

    new_version_conversion = os.path.join(curr_location, "v{}".format(new_version), "dataset_class_v{}_final_{}.txt".format(new_version, new_letter))
    with open(new_version_conversion, 'r', encoding="utf-8") as f:
        # criar e atualizar dicionários
        for line in f:
            line = line.strip()

            if not line:
                continue
            if line[0] == '=':
                break

            # novo ID da classe atual
            classe_atual = line.split()[0]
            new_class_id[classe_atual] = len(new_class_id) + 1

            # lista de classes que compoem a classe atual
            if len(line.split()) > 1:
                used_classes = " ".join(line.split()[1:])
                used_classes = used_classes.strip("[]")
                used_classes = [s.strip() for s in used_classes.split(",")]

                # mapear o ID antigo para o novo de todas as classes que compõem o ID atual
                for classe in used_classes:
                    old_class_id = get_class_id_from_dataset(old_dataset_json, classe)
                    if old_class_id is None:
                        print("Classe {} não encontrada no dataset".format(classe))
                        continue
                    old_id_to_new_id[old_class_id] = new_class_id[classe_atual]
            else:
                # nenhuma classe compõe a classe atual, significa que classe_atual == classe_antiga. atualizar o ID no outro dicionário
                old_class_id = get_class_id_from_dataset(old_dataset_json, classe_atual)
                old_id_to_new_id[old_class_id] = new_class_id[classe_atual]

    classes_removidas, anotacoes_removidas = set(), set()
    # substituir os valores no dataset
    for annotation in old_dataset_json["annotations"]:
        annotation_cid = annotation["category_id"]
        if annotation_cid in old_id_to_new_id:
            annotation["category_id"] = old_id_to_new_id[annotation_cid]
        else:
            # classe foi removida, remover a anotação
            if annotation_cid not in classes_removidas:
                print("Atenção: classe {} foi removida da versão {}.".format(annotation_cid, old_version))
                print("Nome da classe: {}".format([category["name"] for category in old_dataset_json["categories"] if category["id"] == annotation_cid][0]))
                classes_removidas.add(annotation_cid)
            anotacoes_removidas.add(annotation["id"])

    # remover anotações que foram removidas
    new_annotations = [annotation for annotation in old_dataset_json["annotations"] if annotation["id"] not in anotacoes_removidas]
    old_dataset_json["annotations"] = new_annotations

    # atualizar as classes no dataset
    old_dataset_json["categories"] = []
    for key, value in new_class_id.items():
        old_dataset_json["categories"].append({"id": value, "name": key, "supercategory": "food"})

    # remover imagens que não possuem mais classes
    old = json.load(open(old_dataset_location))
    useless_images = get_useless_images(old, classes_removidas)
    new_images = [image for image in old_dataset_json["images"] if image["id"] not in useless_images]
    old_dataset_json["images"] = new_images

    # diretorio para salvar as informações
    new_version_location = os.path.join(curr_location, "v{}".format(new_version), "v{}_{}".format(new_version, new_letter))
    os.makedirs(new_version_location, exist_ok=True)

    # exportar nomes das imagens que foram removidas
    with open(os.path.join(new_version_location, "v{}_{}_images_removed.txt".format(new_version, new_letter)), 'w') as f:
        image_names = [str(image_id).zfill(6) + ".jpg" for image_id in useless_images]
        f.write("\n".join(image_names))

    # exportar dataset
    with open(os.path.join(new_version_location, "v{}_{}_annotations.json".format(new_version, new_letter)), 'w') as f:
        json.dump(old_dataset_json, f)

    print("Dataset v{}_{} criado com sucesso".format(new_version, new_letter))

if __name__ == "__main__":
    create_new_ds_version(0, "O", 1, "A")
    create_new_ds_version(1, "A", 2, "A")
    create_new_ds_version(2, "A", 3, "A")

    create_new_ds_version(0, "O", 1, "M")
    create_new_ds_version(0, "O", 2, "M")
    create_new_ds_version(0, "O", 3, "M")
