# Variational Autoencoders (VAE) y Vector Quantised-Variational Autoencoders (VQ-VAE): Comparación Crítica

## 1. Introducción
Los modelos generativos basados en autoencoders han tenido un papel central en el aprendizaje no supervisado. Entre ellos, los **Variational Autoencoders (VAE)** constituyen un marco probabilístico que permite inferencia aproximada en presencia de variables latentes continuas y distribuciones posteriores intratables. Posteriormente, los **Vector Quantised-Variational Autoencoders (VQ-VAE)** extendieron esta arquitectura hacia representaciones **discretas**, resolviendo algunas limitaciones observadas en los VAE estándar, particularmente el fenómeno conocido como *posterior collapse*.

## 2. Variational Autoencoder (VAE)
El **VAE** combina ideas de modelos gráficos probabilísticos y redes neuronales. Su estructura incluye:
- **Encoder probabilístico**: aproxima la distribución posterior \( q_\phi(z|x) \).
- **Decoder probabilístico**: define \( p_\theta(x|z) \).
- **Objetivo de entrenamiento**: maximizar la *Evidence Lower Bound* (ELBO):

\[
\log p_\theta(x) \geq \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x)\|p_\theta(z)).
\]

### Logros principales
- Introducción de la **reparametrización estocástica** que permite backpropagation eficiente en distribuciones continuas.
- Entrenamiento escalable a grandes datasets mediante **optimización estocástica**.
- Aplicaciones exitosas en visión (denoising, inpainting, generación de dígitos MNIST y rostros de Frey).

### Limitaciones
- **Posterior collapse**: cuando el decoder es muy expresivo (ej. PixelCNN), la variable latente es ignorada, colapsando hacia el prior.
- Representaciones latentes continuas poco interpretables, alejadas de la estructura discreta natural de ciertos datos (lenguaje, fonemas, símbolos).

## 3. Vector Quantised-Variational Autoencoder (VQ-VAE)
El **VQ-VAE** modifica la arquitectura de VAE introduciendo un *cuello de botella discreto* mediante **vector quantisation (VQ)**:
- El encoder produce un vector continuo que es **asignado al embedding más cercano** en un diccionario discreto.
- El decoder recibe el embedding discreto, generando la salida \( p_\theta(x|z) \).
- Se optimiza una función de pérdida que incluye:
  1. Reconstrucción.
  2. *Codebook loss* (para ajustar los embeddings).
  3. *Commitment loss* (para estabilizar la codificación).

### Logros principales
- **Resuelve el posterior collapse** al forzar la utilización de variables discretas.
- Descubrimiento de estructuras discretas de alto nivel:
  - Fonemas en audio sin supervisión.
  - Representaciones simbólicas en imágenes y video.
- Habilitó tareas como **conversión de voz**, **modelado de lenguaje a nivel fonema**, y **generación de video condicional a acciones**.

### Limitaciones
- La cuantización introduce un cuello de botella rígido: la calidad de reconstrucción depende del tamaño del diccionario y de la dimensionalidad de embeddings.
- Entrenamiento más complejo: requiere estrategias como el estimador *straight-through* para propagar gradientes.
- Menor fidelidad visual en reconstrucciones comparado con modelos puramente continuos.

## 4. Diferencias clave entre VAE y VQ-VAE

| Aspecto                  | VAE (2013)             | VQ-VAE (2017)           |
|---------------------------|-----------------------|-------------------------|
| Latentes                  | Continuos (Gaussianos típicamente) | Discretos (via vector quantisation) |
| Problema resuelto         | Inferencia eficiente en modelos con posteriors intratables | Evitar *posterior collapse* y aprovechar variables discretas |
| Limitaciones              | Latentes ignorados con decoders potentes, baja interpretabilidad | Reconstrucciones menos nítidas, dependientes del codebook |
| Logros                    | Escalabilidad, flexibilidad, conexión con autoencoders | Representaciones simbólicas útiles, aplicaciones en audio e imágenes |
| Priors                    | Estático (ej. Gaussiano isotrópico) | Aprendido (ej. PixelCNN, WaveNet)     |

## 5. Conclusión
- El **VAE** representó un avance metodológico crucial al introducir una técnica eficiente de inferencia variacional con variables continuas. Su aporte central fue metodológico y teórico: unificar autoencoders con modelos probabilísticos latentes.
- El **VQ-VAE** construyó sobre esta base resolviendo una limitación práctica crítica (*posterior collapse*) y adaptando el framework a **representaciones discretas**, más naturales para dominios como lenguaje y fonología.
- Ambos modelos son hitos en la evolución de los modelos generativos: el VAE por su elegancia teórica y el VQ-VAE por su eficacia práctica en aplicaciones multimodales.
