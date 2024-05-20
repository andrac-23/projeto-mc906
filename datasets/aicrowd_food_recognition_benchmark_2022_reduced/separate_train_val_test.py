import supervision as sv
import os
import sys
import yaml
import random

def separate_train_val_test(version, letter, percent_train, percent_val, percent_test):
    curr_path = os.path.dirname(os.path.abspath(__file__))
    annotations_directory_path= curr_path + "/v{}".format(version) + "/v{}_{}".format(version, letter) + "/v{}_{}_yoloV8_annotations".format(version, letter)

    annotations = [filename[:-4] for filename in os.listdir(annotations_directory_path)]

    # shuffle annotations
    random.shuffle(annotations)

    used_idxs = set()
    train_annotations, val_annotations, test_annotations = [], [], []

    percent_train, percent_test, percent_val = percent_train/100, percent_test/100, percent_val/100
    # get percent_train% of the annotations for training
    train_annotations_idxs = random.sample(range(len(annotations)), int(percent_train * len(annotations)))
    train_annotations = [annotations[i] for i in train_annotations_idxs]

    used_idxs = set(train_annotations_idxs)
    annotations = [annotations[i] for i in range(len(annotations)) if i not in used_idxs]  # remove training annotations

    # get percent_val% of the remaining annotations for validation
    val_annotations_idxs = random.sample(range(len(annotations)), int(percent_val/(1-percent_train) * len(annotations)))
    val_annotations = [annotations[i] for i in val_annotations_idxs]

    used_idxs = set(val_annotations_idxs)
    annotations = [annotations[i] for i in range(len(annotations)) if i not in used_idxs]  # remove validation annotations

    # the remaining annotations are for testing
    test_annotations = annotations

    assert len(train_annotations) + len(val_annotations) + len(test_annotations) == len(os.listdir(annotations_directory_path)), "Número de anotações incorreto"
    assert len((set(train_annotations) & set(val_annotations)) | (set(train_annotations) & set(test_annotations)) | (set(val_annotations) & set(test_annotations))) == 0, "Anotações repetidas"

    print("Separação realizada com sucesso, verificando classes utilizadas...")

    with open(curr_path + "/v{}".format(version) + "/v{}_{}".format(version, letter) + "/v{}_{}_yoloV8_files".format(version, letter) + "/data.yaml", "r") as file:
        nc = yaml.safe_load(file)["nc"] # número de classes total

    all_annotations = [train_annotations, val_annotations, test_annotations]
    curr_split, nomes = 0, ["train", "val", "test"]
    for split in all_annotations:
        print("Verificando se o split {} ({} imagens) utiliza todas as classes...".format(nomes[curr_split], len(split)))
        if curr_split == 0:
            print("==>Verificação ignorada para o split de treino (provavelmente utilizará todas as classes)\n==>As linhas 49, 50, 51 e 52 do script podem ser comentadas para verificar o split de treino também (demorado)")
            curr_split += 1
            continue
        used_classes = set()
        curr = 1
        for annotation in split:
            print("Progresso atual: {:.2f}%".format(100*curr/len(split)), end="\r")
            curr += 1
            with open(annotations_directory_path + "/" + annotation + ".txt", "r") as file:
                for line in file:
                    used_classes.add(int(line.split()[0]))
        print()
        assert len(used_classes) == nc, "Número de classes incorreto - tente novamente com outro seed"
        print("Split {} utiliza todas as classes".format(nomes[curr_split]))
        curr_split += 1

    print("===> Todos os splits corretos, movendo as respectivas imagens e labels...")

    curr_split = 0
    for split in all_annotations:
        print("Movendo imagens e labels do split {}...".format(nomes[curr_split]))
        curr = 1
        for annotation in split:
            print("Progresso atual: {:.2f}%".format(100*curr/len(split)), end="\r")
            curr += 1
            os.rename(annotations_directory_path + "/" + annotation + ".txt", curr_path + "/v{}".format(version) + "/v{}_{}".format(version, letter) + "/v{}_{}_yoloV8_files".format(version, letter) + "/" + nomes[curr_split] + "/labels/" + annotation + ".txt")
            os.rename(curr_path + "/images/" + annotation + ".jpg", curr_path + "/v{}".format(version) + "/v{}_{}".format(version, letter) + "/v{}_{}_yoloV8_files".format(version, letter) + "/" + nomes[curr_split] + "/images/" + annotation + ".jpg")
        curr_split += 1

    os.rmdir(annotations_directory_path)

    print("Movimentação realizada com sucesso\nFim do script")
    return

if __name__ == "__main__":
    if len(sys.argv) != 6:
	print("Uso incorreto, esperado: python3 separate_train_val_test.py <versao> <letra> <%_train> <%_val> <%_test>")
        sys.exit(1)
    versao, letra = int(sys.argv[1]), sys.argv[2].upper()
    p_train, p_val, p_test = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    if letra not in ["A", "M"] or versao not in [1, 2, 3] or p_train + p_val + p_test != 100:
        print("Argumentos inválidos")
        sys.exit(1)
    separate_train_val_test(versao, letra, p_train, p_val, p_test)
