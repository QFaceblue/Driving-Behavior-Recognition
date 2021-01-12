import requests

resp = requests.post("http://localhost:5000/predict",
                     files={"file": open( r"data/imgs/drink.jpg",'rb')})
print(resp.json())