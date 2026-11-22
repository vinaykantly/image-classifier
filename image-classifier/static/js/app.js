(() => {
  const $ = (id) => document.getElementById(id);
  const MAX_MB = Number(document.querySelector(".wrap").dataset.maxMb);
  const TYPES = ["image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"];
  const el = {
    dz: $("dropzone"), file: $("file-input"), url: $("url-input"), urlBtn: $("url-btn"),
    error: $("error"), loading: $("loading"), result: $("result"), panel: document.querySelector(".panel"),
  };

  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => {
      const on = t === tab;
      t.classList.toggle("active", on);
      t.setAttribute("aria-selected", on);
      $("tab-" + t.dataset.tab).hidden = !on;
    });
    showError("");
  }));

  function showError(msg) { el.error.textContent = msg; el.error.hidden = !msg; }

  function validateFile(f) {
    if (!f) return "Choose an image file first.";
    if (!TYPES.includes(f.type)) return "Unsupported file type. Use a JPG, PNG, WEBP, GIF or BMP image.";
    if (f.size === 0) return "The file is empty.";
    if (f.size > MAX_MB * 1024 * 1024) return `The file is larger than ${MAX_MB} MB.`;
    return "";
  }

  async function classify(formData) {
    showError("");
    el.result.hidden = true;
    el.loading.hidden = false;
    el.urlBtn.disabled = true;
    try {
      const res = await fetch("/api/classify", { method: "POST", body: formData });
      let data = {};
      try { data = await res.json(); } catch (_) { /* non-JSON error page */ }
      if (!res.ok) throw new Error(data.error || "The server could not process that request.");
      render(data);
    } catch (err) {
      showError(err instanceof TypeError ? "Could not reach the server. Is the app still running?" : err.message);
    } finally {
      el.loading.hidden = true;
      el.urlBtn.disabled = false;
    }
  }

  function render(d) {
    $("preview").src = d.image;
    const status = $("status");
    status.classList.toggle("unsure", !d.recognized);
    status.textContent = d.recognized ? "Prediction" : "Not confident enough to name this image";
    $("label").textContent = d.recognized ? d.prediction.label : "Unrecognized";
    $("confidence-block").hidden = !d.recognized;
    if (d.recognized) {
      $("confidence").textContent = d.prediction.confidence.toFixed(1) + "%";
      const fill = $("conf-fill");
      fill.style.width = "0";
      requestAnimationFrame(() => (fill.style.width = d.prediction.confidence + "%"));
    }
    $("explanation").textContent = d.explanation;
    const list = $("others");
    list.replaceChildren();
    d.top_predictions.slice(d.recognized ? 1 : 0).forEach((p) => {
      const li = document.createElement("li");
      const a = document.createElement("span"), b = document.createElement("span");
      a.textContent = p.label; b.textContent = p.confidence.toFixed(1) + "%";
      li.append(a, b);
      list.append(li);
    });
    el.result.hidden = false;
    el.result.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function submitFile(f) {
    const problem = validateFile(f);
    if (problem) return showError(problem);
    const fd = new FormData();
    fd.append("file", f);
    classify(fd);
  }

  el.file.addEventListener("change", () => { submitFile(el.file.files[0]); el.file.value = ""; });
  el.dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.file.click(); } });
  ["dragenter", "dragover"].forEach((ev) => el.dz.addEventListener(ev, (e) => { e.preventDefault(); el.dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => el.dz.addEventListener(ev, (e) => { e.preventDefault(); el.dz.classList.remove("drag"); }));
  el.dz.addEventListener("drop", (e) => submitFile(e.dataTransfer.files[0]));

  function submitUrl() {
    const value = el.url.value.trim();
    if (!value) return showError("Enter an image URL first.");
    if (!/^https?:\/\//i.test(value)) return showError("The URL must start with http:// or https://");
    const fd = new FormData();
    fd.append("url", value);
    classify(fd);
  }
  el.urlBtn.addEventListener("click", submitUrl);
  el.url.addEventListener("keydown", (e) => { if (e.key === "Enter") submitUrl(); });

  $("reset").addEventListener("click", () => {
    el.result.hidden = true; el.url.value = ""; showError("");
    el.panel.scrollIntoView({ behavior: "smooth" });
  });
})();
