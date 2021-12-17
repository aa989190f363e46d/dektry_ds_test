import base64
import requests

ENDPOINT = 'http://127.0.0.1:5000'


def guess(name):

    img_path = f'test_assets/{name}.png'
    with open(img_path, "rb") as img_file:
        img = img_file.read()    
    b64_image = base64.b64encode(img)
    file = {'b64': b64_image}
    r = requests.post(ENDPOINT, file)
    return r.json()['label'][0]


def test_positive():
    for img_name in ('vha8be', 'g6dfdj'):
        assert guess(img_name) == img_name


def test_negative():
    for img_name in ('4k3ul9', 'omybre'):
        assert guess(img_name) != img_name
