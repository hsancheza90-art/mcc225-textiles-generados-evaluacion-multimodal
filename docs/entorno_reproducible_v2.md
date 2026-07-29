# Entorno reproducible OpenCLIP v2

## 1. Entorno canónico

La evaluación v2 utiliza como entorno de referencia:

- Windows 10 de 64 bits;
- Python 3.11;
- PyTorch CPU;
- OpenCLIP `ViT-B-32`;
- pesos `laion2b_s34b_b79k`;
- embeddings de 512 dimensiones;
- contexto textual máximo de 77 tokens.

La ejecución canónica se realiza en CPU para evitar que los resultados dependan de una configuración CUDA específica.

---

## 2. Razón de la ruta externa

El repositorio se encuentra en una ruta extensa de Windows.

Una primera creación de `.venv` dentro del proyecto produjo `WinError 206` durante la instalación de PyTorch debido a una ruta interna demasiado larga.

Por ello se utiliza una ruta corta fuera del repositorio:

    D:\venvs\mcc225-cpu

La ubicación exacta puede cambiar en otro equipo, pero se recomienda evitar rutas profundamente anidadas.

---

## 3. Creación del entorno

Desde PowerShell:

    py -3.11 -m venv D:\venvs\mcc225-cpu

    Set-ExecutionPolicy `
        -Scope Process `
        -ExecutionPolicy Bypass

    & D:\venvs\mcc225-cpu\Scripts\Activate.ps1

    python -m pip install `
        --upgrade `
        pip `
        setuptools `
        wheel

    python -m pip install `
        --no-cache-dir `
        -r requirements.txt

El prompt debe indicar que el entorno `mcc225-cpu` está activo.

El ejecutable de Python puede verificarse con:

    python -c "import sys; print(sys.executable)"

La ruta esperada es:

    D:\venvs\mcc225-cpu\Scripts\python.exe

---

## 4. Dependencias directas

Las dependencias directas están fijadas en `requirements.txt`.

El entorno CPU utiliza:

- `numpy==1.26.4`;
- `pandas==2.2.2`;
- `matplotlib==3.9.2`;
- `Pillow==10.4.0`;
- `torch==2.13.0+cpu`;
- `torchvision==0.28.0+cpu`;
- `open-clip-torch==3.3.0`;
- `scikit-learn==1.5.1`;
- `tqdm==4.66.5`;
- `ipykernel==6.29.5`.

El índice adicional de PyTorch CPU está declarado en el mismo archivo.

---

## 5. Cachés de modelos

Para reducir la longitud de las rutas y evitar descargas dentro del repositorio se emplean cachés externas:

    $env:HF_HOME = "D:\hf-cache\mcc225"
    $env:TORCH_HOME = "D:\torch-cache\mcc225"
    $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

Las carpetas pueden crearse con:

    New-Item `
        -ItemType Directory `
        -Path $env:HF_HOME `
        -Force |
        Out-Null

    New-Item `
        -ItemType Directory `
        -Path $env:TORCH_HOME `
        -Force |
        Out-Null

Los pesos del modelo no se incorporan al repositorio.

La primera ejecución descarga aproximadamente 605 MB para el checkpoint utilizado.

---

## 6. Modelo evaluado

La configuración congelada utiliza:

- biblioteca: `open_clip`;
- arquitectura: `ViT-B-32`;
- pesos: `laion2b_s34b_b79k`;
- dispositivo canónico: CPU;
- dimensión del embedding: 512;
- longitud de contexto: 77 tokens.

El modelo se utiliza en modo de evaluación y sin ajuste adicional:

    model, _, preprocess = (
        open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion2b_s34b_b79k",
        )
    )

    model = model.to("cpu")
    model.eval()

    tokenizer = open_clip.get_tokenizer(
        "ViT-B-32"
    )

---

## 7. Validación del entorno

La coherencia de las dependencias se comprueba con:

    python -m pip check

La auditoría completa se ejecuta mediante:

    python scripts\auditar_entorno_openclip_v2.py

El auditor comprueba:

- versiones de las dependencias directas;
- uso de Python 3.11;
- entorno virtual activo;
- instalación de PyTorch CPU;
- ausencia de CUDA en el runtime canónico;
- configuración del modelo;
- tokenización de 600 textos;
- ausencia de secuencias que ocupen los 77 tokens;
- hashes de configuración y fuentes textuales.

---

## 8. Artefactos del entorno

La auditoría genera:

- `results/v2/entorno_cpu_v2.json`;
- `results/v2/entorno_cpu_pip_freeze.txt`.

`entorno_cpu_v2.json` contiene el resumen estructurado del entorno.

`entorno_cpu_pip_freeze.txt` conserva el inventario completo de paquetes y dependencias transitivas instaladas.

---

## 9. GPU disponible pero no utilizada

El equipo local dispone de una NVIDIA GeForce GTX 1650 con 4 GB de memoria.

Sin embargo, el entorno reproducible utiliza `torch==2.13.0+cpu`. Por tanto:

- `torch.version.cuda` es `None`;
- `torch.cuda.is_available()` devuelve `False`;
- `torch.cuda.device_count()` devuelve `0`;
- la GPU no interviene en los resultados canónicos.

Una ejecución futura con GPU deberá registrarse como una condición experimental diferente y no reemplazar silenciosamente los resultados CPU.

---

## 10. Auditoría del tokenizer

Se auditan tres fuentes textuales:

- 280 captions positivos;
- 40 captions sin color;
- 280 candidatos de negativos difíciles.

En total:

$$
280 + 40 + 280 = 600
$$

textos son procesados por el tokenizer de OpenCLIP.

La salida esperada tiene forma:

    (600, 77)

En la auditoría realizada:

- los captions positivos ocuparon entre 34 y 76 tokens no nulos;
- los captions sin color ocuparon entre 25 y 57 tokens no nulos;
- los candidatos difíciles ocuparon entre 52 y 76 tokens no nulos;
- ninguna secuencia ocupó completamente los 77 tokens.

Por tanto, no se encontró evidencia de truncamiento.

---

## 11. Textos duplicados esperados

De los 600 textos procesados, 494 son únicos.

La diferencia se explica por duplicaciones previstas:

$$
600 - 56 - 50 = 494.
$$

Los 56 candidatos positivos de las consultas repiten los captions canónicos.

Además, 50 negativos difíciles coinciden con captions positivos de otras imágenes:

- 44 por cambio de paleta;
- 6 por cambio de orientación.

Estas repeticiones forman parte del protocolo y no se consideran errores.

---

## 12. Prueba funcional del modelo

La prueba mínima utiliza la imagen `V2_001` y sus cinco candidatos de negativos difíciles.

La ejecución produjo:

- embedding de imagen con forma `(1, 512)`;
- embeddings de texto con forma `(5, 512)`;
- valores de similitud finitos;
- ranking completo de cinco candidatos.

El caption positivo quedó en la posición 2, con margen negativo frente al candidato mejor clasificado:

    rango del positivo: 2
    similitud positiva: 0.300104
    mejor similitud negativa: 0.319497
    margen positivo: -0.019393

Este resultado no indica un error de implementación. Constituye un primer caso de fallo experimental que deberá incorporarse al análisis cualitativo.

---

## 13. Reproducción desde una sesión nueva

Desde la raíz del repositorio:

    Set-ExecutionPolicy `
        -Scope Process `
        -ExecutionPolicy Bypass

    & D:\venvs\mcc225-cpu\Scripts\Activate.ps1

    $env:PYTHONUTF8 = "1"
    $env:HF_HOME = "D:\hf-cache\mcc225"
    $env:TORCH_HOME = "D:\torch-cache\mcc225"
    $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

    python -m pip check

    python scripts\auditar_entorno_openclip_v2.py

El entorno debe mantenerse activo durante la generación de embeddings y la evaluación posterior.

---

## 14. Límites de reproducibilidad

Los resultados numéricos pueden presentar diferencias mínimas entre:

- versiones de procesador;
- bibliotecas matemáticas;
- número de hilos;
- sistemas operativos;
- ejecución CPU y GPU.

Por ello, cada resultado experimental deberá registrar:

- dispositivo;
- versiones de paquetes;
- semilla;
- número de hilos;
- configuración del modelo;
- hashes de las entradas;
- hashes de las salidas.

La reproducibilidad buscada es computacional y auditable, no identidad binaria garantizada entre todo tipo de hardware.