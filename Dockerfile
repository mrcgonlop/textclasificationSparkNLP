ARG IMAGE_VARIANT=slim-buster
ARG OPENJDK_VERSION=8
ARG PYTHON_VERSION=3.6.9

FROM python:${PYTHON_VERSION}-${IMAGE_VARIANT} AS py3
FROM openjdk:${OPENJDK_VERSION}-${IMAGE_VARIANT}

COPY --from=py3 / /

ARG PYSPARK_VERSION=3.1.2
RUN pip --no-cache-dir install spark-nlp==3.4.4 pyspark==${PYSPARK_VERSION} pandas sklearn pyarrow
EXPOSE 8088 8042 4040

WORKDIR testing/test
#COPY test/test.sh test.sh
#RUN ["chmod", "+x", "test.sh"]

# for usage instructions refer to the file docker-compose.yml