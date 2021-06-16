# Aumento de Dados
Aqui contém um guia da organização das moscas por índice. Terá de criar uma pasta ```/datasets``` na diretoria dos ficheiros .py que conterá subdiretorias a referir de seguida. O primeiro índice corresponde ao tipo de manipulação aplicado na imagem, enquanto que os restantes correspondem ao número original das imagens, que estão em  ```/images```. As anotações das imagens originais estão em ```/annots``` em formato xml. Em ```/empty``` estão *sticky traps* donde foram retiradas moscas. Em ```/fly-dataset``` estão moscas retiradas das armadilhas e em ```/gan``` estão as moscas geradas com uma *stylegan-ada* treinada. Finalmente o conjunto final de dados aumentados estã em ```/aug_img``` com as respetivas anotações em ```/aug_xml```. De seguida está referida a indexação de cada uma das armadilhas:

## 1. Manipulação de Imagem Total

- [1]XXXX.jpg: Inversão Vertical

- [2]XXXX.jpg: Inversão Horizontal

- [3]XXXX.jpg: Rotação de 90º

- [4]XXXX.jpg: Rotação de 180º

- [5]XXXX.jpg: Rotação de 270º

## 2. Troca de Moscas entre placas

- [6]XXXX.jpg: Troca de Moscas entre placas

- [7]XXXX.jpg: Substituição de Moscas Originais com Artificiais

## 3. To-Do: Inserção aleatória de Moscas em Sticky Traps

- [8]XXXX.jpg: Inserção de moscas retiradas das sticky traps
- [9}XXXX.jpg: Inserção das moscas geradas pela gan

Dataset disponível em: https://drive.google.com/drive/folders/1S1f_31UuHJiR712RLF7OiqrEicpXF3QD?usp=sharing
