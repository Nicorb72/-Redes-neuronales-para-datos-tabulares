# Publicación en Streamlit Community Cloud

La entrada de la app es `deploy/streamlit_app.py`. Su `requirements.txt` tiene
prioridad sobre el archivo de la raíz y utiliza el paquete PyTorch de CPU.
Los modelos y los datos se cargan desde el repositorio sin reentrenar.

## Configuración del despliegue

- Repositorio: `Nicorb72/-Redes-neuronales-para-datos-tabulares`.
- Rama: `INTEGRANTE-4` mientras se revisa este cambio; `main` al integrarlo.
- Main file path: `deploy/streamlit_app.py`.
- Advanced settings / Python version: `3.11` o `3.12`, versiones de la CI.
- Secrets: ninguno; la app no necesita claves ni APIs externas.
- Compartir como aplicación pública para que pueda abrirla el profesor.

La URL debe obtenerse del despliegue exitoso; no debe suponerse antes de crearlo.
Al publicarla, añadirla al README y al reporte técnico. Comprobar la consulta y
la simulación en una sesión sin autenticar. El enlace al video se configura en
`configs/default.yaml` cuando el equipo lo publique.

Para probar el mismo punto de entrada localmente:

```text
python -m streamlit run deploy/streamlit_app.py
```

Fuente: documentación oficial de organización, dependencias y despliegue de
Streamlit Community Cloud:
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app
