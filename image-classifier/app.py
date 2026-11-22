"""Flask image classification app: upload an image or paste an image URL."""
import base64
import io
import ipaddress
import os
import socket
from urllib.parse import urljoin, urlparse

import requests
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError

from classifier import CONFIDENCE_THRESHOLD, Classifier, explain

MAX_MB = 10
MAX_BYTES = MAX_MB * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "GIF", "BMP"}

Image.MAX_IMAGE_PIXELS = 60_000_000  # guards against decompression bombs

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES + 1024 * 64  # form overhead
classifier = Classifier()


class ApiError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message, self.status = message, status


def assert_public_host(hostname):
    """Block requests to localhost / private networks (SSRF protection)."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        raise ApiError("That web address could not be found. Check the URL and try again.")
    for info in infos:
        if not ipaddress.ip_address(info[4][0]).is_global:
            raise ApiError("That URL points to a private or local address, which is not allowed.")


def fetch_url(url):
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ApiError("Enter a full image URL starting with http:// or https://")
    current = url
    try:
        for _ in range(4):
            assert_public_host(urlparse(current).hostname)
            resp = requests.get(current, stream=True, timeout=(5, 10), allow_redirects=False,
                                headers={"User-Agent": "Mozilla/5.0 ImageClassifier/1.0"})
            if resp.is_redirect and resp.headers.get("Location"):
                current = urljoin(current, resp.headers["Location"])
                resp.close()
                continue
            break
        else:
            raise ApiError("The URL redirected too many times.")
        with resp:
            if resp.status_code != 200:
                raise ApiError(f"The server returned an error ({resp.status_code}) for that URL.")
            ctype = resp.headers.get("Content-Type", "").lower()
            if ctype.startswith("text/"):
                raise ApiError("That URL is a web page, not an image. Use a direct link to an image file.")
            if int(resp.headers.get("Content-Length") or 0) > MAX_BYTES:
                raise ApiError(f"The image is larger than {MAX_MB} MB.", 413)
            data = b""
            for chunk in resp.iter_content(64 * 1024):
                data += chunk
                if len(data) > MAX_BYTES:
                    raise ApiError(f"The image is larger than {MAX_MB} MB.", 413)
            return data
    except requests.exceptions.Timeout:
        raise ApiError("The server took too long to respond.", 504)
    except requests.exceptions.RequestException:
        raise ApiError("Could not download that URL. Check the link and your connection.")


def open_image(data):
    if not data:
        raise ApiError("The file is empty.")
    try:
        probe = Image.open(io.BytesIO(data))
        fmt = probe.format
        probe.verify()
        img = Image.open(io.BytesIO(data))
        img.load()
    except Image.DecompressionBombError:
        raise ApiError("The image has too many pixels to process safely.", 413)
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise ApiError("That file is not a valid image. Use a JPG, PNG, WEBP, GIF or BMP file.", 415)
    if fmt not in ALLOWED_FORMATS:
        raise ApiError(f"{fmt} images are not supported. Use JPG, PNG, WEBP, GIF or BMP.", 415)
    img = ImageOps.exif_transpose(img)
    if img.width < 32 or img.height < 32:
        raise ApiError("The image is too small (minimum 32 x 32 pixels).")
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


def preview_data_url(img):
    thumb = img.copy()
    thumb.thumbnail((900, 900))
    buf = io.BytesIO()
    thumb.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/")
def index():
    return render_template("index.html", max_mb=MAX_MB)


@app.post("/api/classify")
def classify():
    url = (request.form.get("url") or "").strip()
    upload = request.files.get("file")
    if upload and upload.filename:
        data = upload.read()
    elif url:
        data = fetch_url(url)
    else:
        raise ApiError("Choose an image file or enter an image URL first.")
    img = open_image(data)
    results = classifier.predict(img)
    info = explain(results)
    return jsonify({
        "recognized": info["recognized"],
        "explanation": info["message"],
        "prediction": results[0] if info["recognized"] else None,
        "top_predictions": results,
        "threshold": round(CONFIDENCE_THRESHOLD * 100),
        "image": preview_data_url(img),
    })


@app.errorhandler(ApiError)
def handle_api_error(err):
    return jsonify(error=err.message), err.status


@app.errorhandler(413)
def too_large(_):
    return jsonify(error=f"The file is too large. Maximum size is {MAX_MB} MB."), 413


@app.errorhandler(404)
def not_found(_):
    return jsonify(error="Not found."), 404


@app.errorhandler(Exception)
def unexpected(err):
    app.logger.exception(err)
    return jsonify(error="Something went wrong while classifying the image. Please try again."), 500


if __name__ == "__main__":
    print("Loading model (first run downloads the pretrained weights)...")
    classifier.load()
    print("Model ready. Open http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", 5000)), debug=False)
