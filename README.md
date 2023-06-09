# **Predicción de Neumonía**

-----
#### Este proyecto va a consistir en el análisis, exploración de datos y creación de un modelo de red neuronal convolucional para la predicción de una neumonía en un paciente basándonos en una imagen de rayos x. Para ello se han usado diferentes herramientas y librerías enfocadas a las redes neuronales convolucionales.



-----

### Organización de carpetas: 

* scr/
    * data/: Contiene los archivos usados en el proyecto.
    
    * Images/: Contiene imágenes usadas en este archivo Markdown.

    * Model/: Contiene los modelos realizados en el proyecto.

    * notebooks/: son archivos jupyter notebook usados en todo el proceso.

------

### Fuente: [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

------

### En este proyecto de pueden apreciar conocimientos en:

* Python
* Deep Learning
* Keras
* TensorFlow
* ImagineProcessing
* Computer Vision
* Supervised Learning
* Classifier Models

------

## **Importación de los datos**

#### Leemos todas las imágenes de con las que vamos a entrenar a nuestro modelo y que corresponden a imágenes de pacientes sanos. También cambiamos el tamaño de todas las imágenes para que tengan el mismo.

#### Comprobamos que la forma de todas las imágenes es la misma y en blanco y negro ya que si hya algunas a color y otras en blanco y negro nos dará un error al no tener la misma forma todas las imágenes.

#### Hacemos lo mismo de antes con las imágenes de persona con neumonía.
```py
for x in os.listdir("../data/chest_xray/train/PNEUMONIA/"):
    img = imread(os.path.join("../data/chest_xray/train/PNEUMONIA/", x),as_gray=True)
    lista_neumonia.append(cv2.resize(img, (256, 256)))
set_neumonia = set()
for elemento in lista_neumonia:
    set_neumonia.add(elemento.shape)
set_neumonia
output: {(256, 256)}
```
#### Imagen de ejemplo

<img src="src/data/chest_xray/train/NORMAL/IM-0115-0001.jpeg"  width="600" height="500">

## Transformación de datos
#### Transformamos las listas a arrays.

#### Creamos un dataframe con las etiquetas correspondientes a cada imagen y unimos los arrays anteriores.
#### Escalamos los datos.

## Creación de la red neuronal convolucional
#### He elegido crear dos capas de la red convolucional con 2 pooling layers. La activación que he elegido ha sido relu para las capas de entrada y las capas ocultas y sigmoide para la capa de salida ya que va a ser un modelo de clasificación binaria. De optimizador he elegido Adam, de función de pérdida binary_crossentropy ya que es una clasificación binaria y de metrica como función de pérdida accuracy. En la primera capa de la red convolucional aplicamos 64 filtros en un rango de 3x3 píxeles, luego le cambiamos la resolucion. En la tercera capa aplicamos 128 filtros en un rango de 3x3 píxeles y la red convolucional termina con una minimización en su resolución para luego aplanar los datos y meterlos en una red neuronal con 128 neuronas en la capa de entrada y una en la salida ya que queremos que nos devuelva solo un valor binario(1 o 0).

#### Red Convolucional
Tipo de capa|Tamaño de neuronas|Rango del Filtro|FUnción de activación
-----|-----|----|---
Convolucional|65536|3x3|Indefinido
Pooling||2x2|
Convolucional|Indefinido|3x3|Relu
Pooling||2x2|

#### Red Neuronal
Capa| Números de neuronas| Función de activación
------|-----|------
Flatten
1|128|Relu
2|1|Función sigmoide
#### Compilación
Optimizador| Función de coste| Métrica
---|---|---
Adam|Binary Crossentropy| Accuracy

 Layer (type)          |      Output Shape        |      Param #   
---|----|-----
conv2d_4 (Conv2D)      |     (None, 254, 254, 64)  |    640       
max_pooling2d_4 (MaxPooling2D)  | (None, 127, 127, 64)  |   0                                                                      
conv2d_5 (Conv2D)       |    (None, 125, 125, 128)  |   73856     
max_pooling2d_5 (MaxPooling2D)  | (None, 62, 62, 128)  |    0                                                                      
flatten_2 (Flatten)      |    (None, 492032)     |       0         
dense_4 (Dense)       |     (None, 128)      |         62980224  
dense_5 (Dense)       |     (None, 1)       |          129  
## Entrenamiento de la red

#### Barajamos los datos y entrenamos el modelo

#### Elegimos los siguientes parámetros al entrenar el modelo:
Parámetro | Valor
---|----
Batch_size | 128
Épocas | 20
Validation_split | 0,2
## Evaluación del modelo

#### Realizamos todas las operaciones realizadas a los datos pero esta vez a los datos de test y evaluamos el modelo.
Accuracy|0.7371794581413269
----|-----
#### Recordemos que la métrica era accuracy así que su valor es 0,73. es decir, de todas las muestras a predecir, hemos predicho el 73% correctamente.

#### Vamos a ver otras métricas.
Métrica|Valor
----|----
Accuracy| 0.7371794871794872
Recall| 0.9948717948717949
Precission| 0.7054545454545454

#### La métrica Recall nos indica que, de todos los pacientes con neumonia, hemos clasificado bien al 99%. La métrica de Precission nos infica que, de los que hemos predicho como pacientes con neumonía, hemos acertado el 70%. Vemos que nuestro modelo predice muy bien a los pacientes con neumonia pero también hay una cantidad considerable de falsos positivos.
