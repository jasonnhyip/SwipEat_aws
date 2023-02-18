
# Welcome to SwipEat project!

SwipEat aims to provide a list of nearby restaurants based on the user's profile and history. It is an Android/iOS mobile application that is built using Flutter, AWS Lambda and Firebase Realtime Database.

This repository contains the backend part of SwipEat.

## Build Instruction

Here is the instruction on how to implement the AWS Lambda Function with Docker.

Linux OS is required. Ubuntu 20.04 is preferred.


Prepare the packages
```
$ sudo apt-get update
$ sudo apt-get -y install python3-pip
$ sudo apt-get install docker.io
$ sudo apt-get install curl   #for local testing and package downloading
$ python3 -m pip install aws-cdk-lib
```

Install AWS RIE for local testing and debugging

```
$ mkdir -p ~/.aws-lambda-rie && curl -Lo ~/.aws-lambda-rie/aws-lambda-rie https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie && chmod +x ~/.aws-lambda-rie/aws-lambda-rie
```

Copy this repository to desired destination.

```
$ cd aws_lambda_swipeat
```

Activate virtualenv and install required dependencies.

```
$ source .venv/bin/activate
$ pip install -r requirements.txt && pip install -r requirements-dev.txt
```

Build and run docker image in localhost

```
$ cd aws_lambda_swipeat/aws_lambda_swipeat/DockerLambda/
$ sudo chmod 666 /var/run/docker.sock
$ sudo service docker restart
$ docker build -t aws_lambda_swipeat:latest .
$ docker run -p 9000:8080 aws_lambda_swipeat:latest
```

Test with curl. Alternative: Postman desktop app

```
$ curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{"command": "predict" ,"uid":"ATGno7897veXPzgTueLkR7p3ncv2", "latitude": 22.31300456133964, "longitude": 114.22626022773215, "price": 1, "yelpCategories": "[\"british\",\"cantonese\",\"indpak\",\"hotdogs\"]"}'


```

## Deploy to AWS Lambda

```
$ curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
$ unzip awscliv2.zip
$ sudo ./aws/install
$ cd aws_lambda_swipeat
$ aws configure #configure with the AWS account credentials
$ cdk deploy
```

Enjoy!


# Project Overview

## Data Collection:

### Restaurant Dataset:
1. Fetch the data of restaurants in nearby manner via Yelp Fusion API - Business Search Endpoint, Business Details Endpoint

2. Generate equally spaced locations (longitude and latitude) in HK
<img src="/images/dots_kl_nt.png" width=50% height=50%>

3. Information about 7267 restaurants are collected

### Restaurant Rating Dataset:
Questionnare with 5 questions published on qualtrics:
- Gender, Age, Occupation, Dining Preferences, Ratings of the visited restaurants

109 responses and 1153 restaurant ratings were collected


## System Design:
<img src="/images/aws overview.drawio.png" width=100% height=100%>

### Logic Flow of a Lambda Handler:
<img src="/images/logic_flow.drawio.png" width=100% height=100%>

### System Structure for Recommendation Mode:
<img src="/images/nn.drawio.png" width=100% height=100%>

### Model Tuning Result:
<img src="/images/model_tuning_result.png" width=100% height=100%>

## Demo

https://user-images.githubusercontent.com/44663265/214591958-cd144348-00dd-43da-b573-980f84eb56a3.mp4


