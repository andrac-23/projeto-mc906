## Como criar os arquivos do YOLOv8 usando os arquivos dataset_class_v{}\_final_{}.txt
---
- ### Etapa 0: Requisitos:

1. Baixar [[Round 2] public_training_set_release_2.1.tar.gz (3.1G)](https://www.aicrowd.com/challenges/food-recognition-benchmark-2022/dataset_files) (entrar pelo Google se não tiver conta)

2. Clonar o repositório e copiar o diretório atual (`aicrowd_food_recognition_benchmark_2022_reduced`) para algum novo diretório do computador

3. Extrair os arquivos do `.zip` no diretório atual

4. Renomear o arquivo `annotations.json` extraído para `v0_O_annotations.json`, e movê-lo para a pasta `v0/v0_O`

5. Abrir o terminal no diretório atual e executar o comando `pip install -r requirements.txt`

 - ### Etapa 1: Criação dos novos arquivos annotations.json:
 > **Essa etapa só precisa ser executada uma vez**
 >  Os novos annotations.json serão chamados de `annotations_v{}_{}.json` e serão utilizados para obtermos os nomes das imagens de cada classe e as coordenadas (em px) das bounding boxes

 1. Executar o comando `python3 combine_classes_script.py` (~1GB de arquivos serão criados)

 2. Caso uma classe seja removida na versão criada do dataset_class, os arquivos chamados `v{}_{}_images_removed.txt` terão os nomes das imagens que não serão mais necessárias (não terão mais nenhuma anotação) - talvez essas informações sejam utilizadas no futuro

- ### Etapa 2: Criação dos arquivos do YOLOv8 para uma versão específica do `dataset_class_v{}\_final_{}.txt`:
> Após escolher a versão e letra do dataset (ex: 2 e A), essa etapa criará as respectivas anotações (labels) no formato do YOLOv8, criará o arquivo data.yaml, dividirá o dataset em treino/validação/teste e moverá as imagens e labels para os diretórios específicos

1. Executar o comando `python3 create_yolo_labels_and_dirs.py <versao> <letra>` para criar as labels no formato do YOLOv8 e a estrutura dos diretórios do YOLO. Ex: `python3 create_yolo_labels_and_dirs.py 1 A`.

2. Executar o comando `python3 separate_train_val_test.py <versao> <letra> <%_treino> <%_validacao> <%_teste>` para criar a separação do dataset em treino/validação/teste usando as porcentagens dadas (pode demorar um pouco). Ex: `python3 separate_train_val_test.py 1 A 80 10 10`.

3. Agora o diretório `v{}/v{}\_{}/v{}_{}_yoloV8files` está pronto para ser treinado pelo YOLOv8. As imagens e as labels foram copiadas para as respectivas pastas. É possível criar um .zip desse diretório e subi-lo no Colab, por exemplo.

4. Para gerar esses arquivos para outras versões do dataset, basta escolher a nova versão e letra. As etapas 0 e 1 não precisam ser executadas novamente.

- ### Etapa 3: K-fold
> TODO
