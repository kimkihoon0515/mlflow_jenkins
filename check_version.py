from mlflow.tracking import MlflowClient
client = MlflowClient(tracking_uri='http://13.125.54.121:5000')

import pymysql
conn = pymysql.connect(host="13.125.54.121",port=3306,user="root",password="password",db="bento",charset="utf8")
cur = conn.cursor()
import bentoml
import numpy as np
import subprocess



def check_latest_version(name):
    models = client.get_latest_versions(name,stages=['Production'])
    version = 0

    for mv in models:
        version = mv.version

    version = int(version)
    return version


def check_serving_version(name):
    s = f"SELECT version FROM bento.serving where name = '{name}';"
    try:
        cur.execute(s)
        data = cur.fetchone()
        version = data[0]
        version = int(version)
    except:
        return False
    
    return version

def db_insert(name,version):
    s = f"INSERT INTO bento.serving (name,version) VALUES ('{name}',{version});"
    cur.execute(s)
    conn.commit()

def db_update(name,version):
    s = f"update bento.serving set version = {version} where name = '{name}';"
    cur.execute(s)
    conn.commit()
    


if __name__ == "__main__":

    global name
    name = 'basic_model'

    if not check_serving_version(name):
        db_insert(name,check_latest_version(name))
        print("DB에 버전 모델과 버전 등록 완료")

    else:
        if check_latest_version(name) != check_serving_version(name):
            print(f"최신 Production 버전은 {check_latest_version(name)} 이고 현재 Serving 중인 버전은 {check_serving_version(name)}입니다.")
            db_update(name,check_latest_version(name))
            print("")
            subprocess.call("python3 hello.py",shell=True)
            print(f"최신 Production 버전은 {check_latest_version(name)} 이고 현재 Serving 중인 버전은 {check_serving_version(name)}입니다.")


        else:
            print(f"현재 Serving 중인 버전은 {check_serving_version(name)} 입니다.")
            subprocess.call("python3 hello.py",shell=True)




