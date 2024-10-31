import requests

def fetch_data(api, payload, debug=False):
    
    response = requests.get(api, params = payload)

    if not response.ok:
        raise Exception("API response not ok: ", response.status_code, response.text)
    
    if debug:
        print(response.url)
        print(response.status_code)
        print(response.text)

    return response.json()