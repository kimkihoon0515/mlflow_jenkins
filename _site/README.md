# AWS to Jenkins Practice

# Pipeline

## 1. sh train.sh -> mlflow에 새로운 버전 생성
## 2. 13.125.54.121:5000 에 접속 후 모델 Stage를 Production으로 변경
## 3. Jenkins가 일정 주기마다 check version.py를 실행해서 현재 Serving 중인 버전과 새롭게 올라온 버전비교
## 4-1. 두 버전의 차이가 있다면 bentoml build를 통해 service:svc를 실행하여 새로운 모델 버전을 update하고 db에 버전을 update 시킨다. 그 후 bentoml docker image를 만들어낸다.
## 4-2 두 버전의 차이가 없다면 현재 Serving 중인 모델을 계속 유지한다.