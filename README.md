# Diplodatos 2020 - Aprendizaje Profundo
## Practico de la Materia

## Integrantes

- Leonardo Palacios
- Marcelo Tisera
- Dario Yvanoff
- Victoria Santucho
- Guillermo Adolfo Coseani

## Estructura del GIT
```
|-experiment
||-cnn_experiments.py
||-cnn.py
||-mlp.py
||-*
|-mlruns
|-Practico.md
|-README.md
|-run_cnn_experiments.sh      <-Ejecuta experiment/cnn_experiments.py
|-run_cnn.sh                  <-Ejecuta experiment/cnn.py
|-run_mlp.sh                  <-Ejecuta experiment/mlp.py

```
Los archivos `run_mlp.sh`, `run_cnn.sh` y `run_cnn_experiments.sh` han sido adaptados del original `run.sh` para ejecutar diferentes tipos de modelos contenidos en sus archivos homologs dentro de la carpeta `experiment`.

Ademas, dentro de la carpeta `experiment` se encontraran archivos `*_debug.py` utilizados para debugear el codigo y tener un mejor entendimiento de que se hacía paso a paso.

Para entrenar cualquiera de los modelos del repositorio basta con estar dentro del environment `deeplearning` de conda y ejecuar el archivo `.sh` elegido.

## Experimentos
#### MLP `(diplodatos.spanish.MLP)`
Sobre el modelo mlp.py se probaron distintos valores para algunos de los hiperparametros:
* Dropout = [0.1, 0.3]. Los resultados mejoraron al tomar el 0.1.

* Batch_size = [128, 64, 356]. En este caso, los resultados solo difieren en la décima o centésima, por lo que decidimos quedarnos
con el valor 128.
* random_buffer_size = [2048, 1024, 3072, 4096]. Los resultados tienen una leve mejora con el valor 4096, sin embargo lo considerable es la
reducción en el tiempo de cómputo, por lo que nos quedamos con este valor (4096).
* x = [torch.mean() , torch.sum()] En este caso probamos con torch.sum(), los resultados no mejoraron, por lo que mantuvimos
el torch.mean()
* freezen_embedings = [True, False]. Probamos con no freezar los embeddings. Los resultados mejoraron considerablemente, por consiguiente se
decidió freezarlos. El tiempo de cómputo también aumenta considerablemente, sin embargo los resultados lo justifican.
* Hidden_layers =[[256, 128] [512, 128] , [512, 256] ]. Se encontró que a medida que aumentabamos el tamaño de las capas, los resultados mejoraban. Se decidio
tomar para el modelo final el tamaño [512, 256].

#### CNN `(diplodatos.spanish.CNN)`

En el caso de CNN se realizaron experimentos con distintos hiper parámetros y en una combinación de entrenamiento realizadas
en notebooks de los integrantes del grupo y en Nabuco. En este ultimo caso se utilizo GPU encolando los jobs con slurm.
Los resultados en principio no fueron buenos aunque fueron mejorando a medida que se cambiaron algunos hiper parametros.
Concretamente en algunos experimentos se cambiaron la cantidad de epocs de 3 a 5. Con 5 epocs se probo en Nabuco y
mejoraron levemente los resultados. Cabe aclarar que no se realizaron experimentos con mas epocs para poder compartir
los recursos de Nabuco con otros grupos. Tambien se noto que cambiando el hiper parametro de dropouts mejoraba a medida
que se decrementaba su valor y empeoraba naturalmente cuando se incrementaba dicho valor.
En este repo se subieron los scripts con los mejores hiper parametros encontrados en todos los experimentos.
Se utilizó mlflow para dejar registro de todos los experimentos realizados en las distintas notebooks y en Nabuco.

#### CNN+MLP `(diplodatos.spanish.CNN.Experiments)`
Por ultimo, hicimos algunas pruebas con una red CNN adjuntandole en sus ultimas capas una MLP. Tambien aprovechamos para experimentar cambiandole la funcion de activacion de la ultimas capas (`tanh` en vez de `relu`), pero sin evidenciar grandes cambios en los valores de bacc y loss.
