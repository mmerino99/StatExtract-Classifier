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

  const warningBanner = document.getElementById("warningBanner");
  const warningMsg    = document.getElementById("warningMsg");
  const resultBadge   = document.getElementById("resultBadge");
  const resultSummary = document.getElementById("resultSummary");
  const confidenceBar = document.getElementById("confidenceBar");
  const confidencePct = document.getElementById("confidencePct");
  const allScoresEl   = document.getElementById("allScores");
  const chipEngine    = document.getElementById("chipEngine");
  const chipLang      = document.getElementById("chipLang");
  const chipWords     = document.getElementById("chipWords");
  const chipTime      = document.getElementById("chipTime");
  const textPreview   = document.getElementById("textPreview");
  const errorMessage  = document.getElementById("errorMessage");
  const resetBtn      = document.getElementById("resetBtn");
  const errorResetBtn = document.getElementById("errorResetBtn");
  const loadingStage  = document.getElementById("loadingStage");
  const invoiceFieldsBlock = document.getElementById("invoiceFieldsBlock");
  const invNumber = document.getElementById("invNumber");
  const invDate = document.getElementById("invDate");
  const invDueDate = document.getElementById("invDueDate");
  const invIssuer = document.getElementById("invIssuer");
  const invRecipient = document.getElementById("invRecipient");
  const invTotal = document.getElementById("invTotal");

  let selectedFileObj = null;
  let _startTime      = 0;

  // ── Utility ──────────────────────────────────────────────────────────────
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
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) setFile(file);
  });
  dropZone.addEventListener("click", (e) => {
    if (e.target === browseBtn) return;
    fileInput.click();
  });
  browseBtn.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
  fileInput.addEventListener("change", () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });
  clearBtn.addEventListener("click", (e) => { e.stopPropagation(); clearFile(); });

  // ── Classify ─────────────────────────────────────────────────────────────
  classifyBtn.addEventListener("click", classify);

  // Animate the loading stage label to show progress phases
  let _loadInterval = null;
  const STAGES = ["Running OCR…", "Vectorising text (TF-IDF)…", "Classifying (SVM)…", "Finalising…"];
  function startLoadingAnimation() {
    let i = 0;
    loadingStage.textContent = STAGES[0];
    _loadInterval = setInterval(() => {
      i = Math.min(i + 1, STAGES.length - 1);
      loadingStage.textContent = STAGES[i];
    }, 2500);
  }
  function stopLoadingAnimation() {
    clearInterval(_loadInterval);
  }

  async function classify() {
    if (!selectedFileObj) return;

    showCard(loadingCard);
    startLoadingAnimation();
    _startTime = performance.now();

    const form = new FormData();
    form.append("file", selectedFileObj);

    try {
      const res  = await fetch("/classify", { method: "POST", body: form });
      const data = await res.json();

      if (!res.ok) throw new Error(data.error || `Server error ${res.status}`);
      renderResult(data);

    } catch (err) {
      renderError(err.message || "Unknown error");
    } finally {
      stopLoadingAnimation();
    }
  }

  // ── Render result ────────────────────────────────────────────────────────
  function renderResult(data) {
    const label      = (data.label || "UNKNOWN").toUpperCase();
    const confidence = data.confidence ?? 0;

    // ── Badge ──
    resultBadge.textContent = label;
    resultBadge.setAttribute("data-label", label);
    resultBadge.classList.toggle("uncertain", !!data.is_uncertain);

    // ── Summary ──
    resultSummary.textContent = data.summary || `This is a ${label}.`;

    // ── Warning banner ──
    if (data.warning && data.warning_msg) {
      warningMsg.textContent = data.warning_msg;
      warningBanner.classList.remove("hidden");
      warningBanner.classList.toggle("warning-banner--error", !!data.is_non_english);
    } else {
      warningBanner.classList.add("hidden");
      warningBanner.classList.remove("warning-banner--error");
    }

    // ── Primary confidence bar (colour-coded) ──
    confidencePct.textContent = `${confidence}%`;
    confidenceBar.style.width = "0";
    confidenceBar.style.background =
      confidence >= 70 ? "var(--conf-high)" :
      confidence >= 45 ? "var(--conf-mid)"  :
                         "var(--conf-low)";

    // ── All class scores mini chart ──
    allScoresEl.innerHTML = "";
    (data.all_scores || []).forEach((s, idx) => {
      const isTop = idx === 0;
      const row   = document.createElement("div");
      row.className = "score-row";
      row.innerHTML = `
        <span class="score-label">${s.label}</span>
        <div class="score-bar-bg">
          <div class="score-bar-fill${isTop ? " top" : ""}"
               style="width:0"
               data-target="${s.confidence}"></div>
        </div>
        <span class="score-pct${isTop ? " top" : ""}">${s.confidence}%</span>`;
      allScoresEl.appendChild(row);
    });

    // ── Diagnostics chips ──
    const engineLabel = data.ocr_engine === "easyocr" ? "EasyOCR (handwriting)" : "Tesseract";
    chipEngine.textContent  = "";
    chipEngine.innerHTML    = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg> OCR: ${engineLabel}`;
    chipEngine.className    = `diag-chip engine-${data.ocr_engine || "tesseract"}`;

    const langName  = data.language_name || data.detected_language || "Unknown";
    const langClass = data.is_non_english ? "lang-non-english" : "lang-english";
    chipLang.innerHTML  = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg> ${langName}`;
    chipLang.className  = `diag-chip ${langClass}`;

    chipWords.innerHTML     = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M14 2H6a2 2 0 0 0-2 2v16h16V8z"/><polyline points="14 2 14 8 20 8"/></svg> ${data.word_count ?? "—"} words extracted`;
    chipTime.innerHTML      = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg> ${data.elapsed_ms ?? "—"} ms`;

    // ── Invoice fields (invoice-only) ──
    const inv = data.invoice_fields;
    const showInv = label === "INVOICE" && inv && typeof inv === "object";
    if (invoiceFieldsBlock) {
      invoiceFieldsBlock.classList.toggle("hidden", !showInv);
    }
    if (showInv) {
      invNumber.textContent = inv.invoice_number ?? "—";
      invDate.textContent = inv.invoice_date ?? "—";
      invDueDate.textContent = inv.due_date ?? "—";
      invIssuer.textContent = inv.issuer_name ?? "—";
      invRecipient.textContent = inv.recipient_name ?? "—";
      invTotal.textContent = (inv.total_amount ?? "—").toString();
    }

    // ── Text preview ──
    textPreview.textContent = data.text_preview || "(no text extracted)";

    // ── Show card, then animate bars ──
    showCard(resultCard);
    requestAnimationFrame(() => requestAnimationFrame(() => {
      confidenceBar.style.width = `${confidence}%`;
      document.querySelectorAll(".score-bar-fill[data-target]").forEach((el) => {
        el.style.width = `${el.dataset.target}%`;
      });
    }));
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
