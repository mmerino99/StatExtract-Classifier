/* app.js – drag-and-drop classifier frontend */

(function () {
  "use strict";

  // ── Element references ──────────────────────────────────────────────────
  const dropZone      = document.getElementById("dropZone");
  const fileInput     = document.getElementById("fileInput");
  const browseBtn     = document.getElementById("browseBtn");
  const selectedFile  = document.getElementById("selectedFile");
  const fileName      = document.getElementById("fileName");
  const clearBtn      = document.getElementById("clearBtn");
  const classifyBtn   = document.getElementById("classifyBtn");

  const uploadCard    = document.getElementById("uploadCard");
  const loadingCard   = document.getElementById("loadingCard");
  const resultCard    = document.getElementById("resultCard");
  const errorCard     = document.getElementById("errorCard");

  const resultBadge   = document.getElementById("resultBadge");
  const resultSummary = document.getElementById("resultSummary");
  const confidenceBar = document.getElementById("confidenceBar");
  const confidencePct = document.getElementById("confidencePct");
  const textPreview   = document.getElementById("textPreview");
  const warningBanner = document.getElementById("warningBanner");
  const warningMsg    = document.getElementById("warningMsg");
  const errorMessage  = document.getElementById("errorMessage");

  const resetBtn      = document.getElementById("resetBtn");
  const errorResetBtn = document.getElementById("errorResetBtn");

  let selectedFileObj = null;

  // ── State helpers ───────────────────────────────────────────────────────
  function showCard(card) {
    [uploadCard, loadingCard, resultCard, errorCard].forEach((c) => {
      c.classList.toggle("hidden", c !== card);
    });
  }

  function setFile(file) {
    selectedFileObj = file;
    fileName.textContent = file.name;
    selectedFile.classList.remove("hidden");
    classifyBtn.disabled = false;
  }

  function clearFile() {
    selectedFileObj = null;
    fileInput.value = "";
    selectedFile.classList.add("hidden");
    classifyBtn.disabled = true;
  }

  function reset() {
    clearFile();
    showCard(uploadCard);
  }

  // ── Drop zone events ────────────────────────────────────────────────────
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) setFile(file);
  });

  // clicking anywhere in the drop zone opens the file picker
  dropZone.addEventListener("click", (e) => {
    if (e.target === browseBtn) return;   // handled separately below
    fileInput.click();
  });

  browseBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) setFile(fileInput.files[0]);
  });

  clearBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    clearFile();
  });

  // ── Classify ─────────────────────────────────────────────────────────────
  classifyBtn.addEventListener("click", classify);

  async function classify() {
    if (!selectedFileObj) return;

    showCard(loadingCard);

    const form = new FormData();
    form.append("file", selectedFileObj);

    try {
      const res  = await fetch("/classify", { method: "POST", body: form });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || `Server error ${res.status}`);
      }

      renderResult(data);

    } catch (err) {
      renderError(err.message || "Unknown error");
    }
  }

  // ── Render result ────────────────────────────────────────────────────────
  function renderResult(data) {
    const label = (data.label || "UNKNOWN").toUpperCase();
    resultBadge.textContent        = label;
    resultBadge.setAttribute("data-label", label);
    resultSummary.textContent      = data.summary || `This is a ${label}.`;
    confidencePct.textContent      = `${data.confidence}%`;
    textPreview.textContent        = data.text_preview || "(no text extracted)";

    // Show or hide the low-quality warning banner
    if (data.warning && data.warning_msg) {
      warningMsg.textContent = data.warning_msg;
      warningBanner.classList.remove("hidden");
    } else {
      warningBanner.classList.add("hidden");
    }

    // Animate confidence bar after card is visible
    confidenceBar.style.width = "0";
    showCard(resultCard);
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        confidenceBar.style.width = `${data.confidence}%`;
      });
    });
  }

  // ── Render error ─────────────────────────────────────────────────────────
  function renderError(msg) {
    errorMessage.textContent = msg;
    showCard(errorCard);
  }

  // ── Reset buttons ─────────────────────────────────────────────────────────
  resetBtn.addEventListener("click", reset);
  errorResetBtn.addEventListener("click", reset);

})();
