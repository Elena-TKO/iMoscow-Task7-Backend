import requests

url = "http://localhost:8000/api/analyze-image"

try:
    files = {"image": ("tree.jpg", open("./data/images/image2.jpeg", "rb"), "image/jpeg")}

    response = requests.post(url, files=files)

    print(f"Status: {response.status_code}")
    print(f"Headers: {response.headers}")
    print(f"Content-Type: {response.headers.get('content-type')}")
    print(f"Text: {response.text[:500]}")

except Exception as e:
    print(f"Error: {e}")
finally:
    if "files" in locals():
        files["image"][1].close()
