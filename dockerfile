FROM jupyter/datascience-notebook

# RUN pip install required.txt
RUN mkdir app
WORKDIR /app
EXPOSE 8888

#pagina 6