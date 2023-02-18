FROM amazon/aws-lambda-python:latest

LABEL maintainer="u3590650@connect.hku.hk"
# Installs python, removes cache file to make things smaller
RUN yum update -y && \
    yum install -y python3 python3-dev python3-pip gcc && \
    rm -Rf /var/cache/yum
# Be sure to copy over the function itself!
COPY docker_lambda.py ./
COPY model.py ./
COPY firebase.py ./
COPY yelp.py ./
COPY helper.py ./
# Copies requirements.txt file into the container
COPY requirements.txt ./
# Installs dependencies found in your requirements.txt file
RUN pip install -r requirements.txt
RUN mkdir -p /mnt/models
# Points to the handler function of your lambda function
CMD ["docker_lambda.handler"]
