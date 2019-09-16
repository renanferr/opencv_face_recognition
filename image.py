import requests
# import asyncio


def recognize(b64):
    print("Sending image...")
    data = {'photo': b64}
    r = requests.post('https://facial-garagem-backend-welcome.azurewebsites.net/detect-face', data=data)
    return r.json()
