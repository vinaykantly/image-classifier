# Image Classifier (Flask + PyTorch)

Upload an image or paste an image URL and a pretrained deep learning model
(MobileNetV2 by default, ResNet50 optional) identifies what is in it. No login,
no API keys, no paid services. The model runs locally on your machine.

## Folder structure
```
image-classifier/
├── app.py              Flask server, upload/URL handling, validation, errors
├── classifier.py       Pretrained model loading, prediction, explanation
├── requirements.txt
├── README.md
├── templates/index.html
└── static/
    ├── css/style.css
    └── js/app.js
```

## Run in VS Code
1. Install Python 3.9 - 3.12 (3.12 recommended) from python.org. Tick "Add Python to PATH" on Windows.
2. Open the `image-classifier` folder in VS Code (File > Open Folder).
3. Open a terminal (Terminal > New Terminal) and create a virtual environment:
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv venv` then `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
   (PyTorch is a large download. CPU-only is fine for this app.)
5. Start the app: `python app.py`
   The first run downloads the pretrained weights (~14 MB for MobileNetV2) and
   caches them. Internet is needed for that one time only.
6. Open http://127.0.0.1:5000 in your browser.

If VS Code shows the wrong interpreter, press Ctrl+Shift+P > "Python: Select Interpreter" and pick the `venv` one.

## Settings (optional environment variables)
| Variable | Default | Meaning |
|---|---|---|
| `MODEL_NAME` | `mobilenet_v2` | `mobilenet_v2` (fast) or `resnet50` (~100 MB, more accurate) |
| `CONFIDENCE_THRESHOLD` | `0.30` | Below this top-1 probability the app says it is not confident |
| `PORT` | `5000` | Server port |

Example (PowerShell): `$env:MODEL_NAME="resnet50"; python app.py`
Example (macOS/Linux): `MODEL_NAME=resnet50 python app.py`

## How it works
- Images are validated with Pillow (real format check, size limit 10 MB, minimum 32x32, pixel-count limit).
- The image is preprocessed and fed to a torchvision model trained on ImageNet (1,000 classes).
- Softmax gives probabilities; the top 5 are returned. If the best is under the threshold, the app refuses to name the image.
- For URLs, the server downloads the image with a timeout and size cap and refuses localhost/private network addresses.

## Limitations
- The model only knows the 1,000 ImageNet categories. Things outside them (for example most logos, text, or specific people) get low confidence or a nearby label.
- ImageNet has no general "person" class, so photos of people are often labelled by clothing, a role or the surroundings.
- Some websites block automated downloads. If a URL fails, save the image and upload it instead.

## Troubleshooting
- `pip install torch` fails: check `python --version` is 3.9 - 3.12, then upgrade pip with `python -m pip install --upgrade pip`.
- Weights fail to download: check your internet connection and any proxy or firewall, then rerun.
- Port in use: set `PORT=5001` and open that port instead.
