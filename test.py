
import requests

base_url = "http://127.0.0.1:5000/"
data = {"sentence": "ok no t dret"}
post = requests.post(f"{base_url}/translate/cv/en", data)
print(dir(post))
print(post.text)