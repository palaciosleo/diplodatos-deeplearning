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
