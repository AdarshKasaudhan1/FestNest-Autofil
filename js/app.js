/**
 * FestNest AutoFill — Client Application Script
 * Brand: festnest.in
 * Connects to: GET /api/samples, GET /api/preview/<filename>, POST /api/autofill
 */

(() => {
  'use strict';

  // ==========================================================================
  // Configuration
  // ==========================================================================
  const CONFIG = {
    maxMB: 25,
    timeoutMs: 240000,          // 240 seconds hard timeout
    maxConcurrentThumbs: 3,     // Max concurrent /api/preview requests
    staggerMs: 70,              // Staged fill animation delay per field
    sampleDebounceMs: 180,
  };

  const SCHEMA_FIELDS = [
    'event_name',
    'event_type',
    'date',
    'time',
    'venue',
    'organizer',
    'prize_pool',
    'registration_deadline',
    'contact',
    'description',
  ];

  const STANDARD_CATEGORIES = [
    'Hackathon',
    'Cultural Fest',
    'Workshop',
    'Competition',
    'Tech Talk',
    'Sports',
    'Mega Fest',
  ];

  // DOM Helper
  const $ = (id) => document.getElementById(id);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  // ==========================================================================
  // Application State
  // ==========================================================================
  let activeAbortCtrl = null;
  let activeJob = null;
  let originalData = {};
  let currentPosterBlobUrl = null;
  let currentPreviewUrl = '';
  let allSamples = [];
  let busyTimer = null;
  let busyStartTime = 0;
  let toastTimer = null;

  // ==========================================================================
  // Concurrency-Limited Thumbnail Queue
  // ==========================================================================
  class ThumbnailQueue {
    constructor(concurrency = 3) {
      this.concurrency = concurrency;
      this.running = 0;
      this.queue = [];
    }

    add(task) {
      this.queue.push(task);
      this.process();
    }

    process() {
      while (this.running < this.concurrency && this.queue.length > 0) {
        const task = this.queue.shift();
        this.running++;
        task().finally(() => {
          this.running--;
          this.process();
        });
      }
    }
  }

  const thumbQueue = new ThumbnailQueue(CONFIG.maxConcurrentThumbs);

  // ==========================================================================
  // UI State Switcher
  // ==========================================================================
  function setUiState(state) {
    const states = ['idle', 'busy', 'error', 'result'];
    states.forEach((s) => {
      const el = $(`state-${s}`);
      if (el) el.hidden = s !== state;
    });

    if (state === 'busy') {
      startBusyProgress();
    } else {
      stopBusyProgress();
    }
  }

  // ==========================================================================
  // Stepper & Busy Timer
  // ==========================================================================
  function startBusyProgress() {
    stopBusyProgress();
    busyStartTime = Date.now();
    const elapsedEl = $('busy-elapsed');
    const headlineEl = $('busy-headline');
    const slowHintEl = $('busy-slow-hint');
    const step1 = $('step-1');
    const step2 = $('step-2');
    const step3 = $('step-3');

    if (slowHintEl) slowHintEl.hidden = true;

    const updateTick = () => {
      const elapsedSec = (Date.now() - busyStartTime) / 1000;
      if (elapsedEl) elapsedEl.textContent = `${elapsedSec.toFixed(1)}s`;

      if (elapsedSec < 2.0) {
        if (headlineEl) headlineEl.textContent = 'Reading poster…';
        setStepActive(step1, [step2, step3]);
      } else if (elapsedSec < 6.5) {
        if (headlineEl) headlineEl.textContent = 'Extracting text & layout…';
        setStepDone(step1);
        setStepActive(step2, [step3]);
      } else {
        if (headlineEl) headlineEl.textContent = 'Refining event fields…';
        setStepDone(step1);
        setStepDone(step2);
        setStepActive(step3, []);
      }

      if (elapsedSec >= 15.0 && slowHintEl) {
        slowHintEl.hidden = false;
      }
    };

    updateTick();
    busyTimer = setInterval(updateTick, 100);
  }

  function stopBusyProgress() {
    if (busyTimer) {
      clearInterval(busyTimer);
      busyTimer = null;
    }
  }

  function setStepActive(activeStep, inactiveSteps) {
    if (activeStep) {
      activeStep.classList.add('active');
      activeStep.classList.remove('done');
    }
    inactiveSteps.forEach((s) => {
      if (s) {
        s.classList.remove('active');
        s.classList.remove('done');
      }
    });
  }

  function setStepDone(step) {
    if (step) {
      step.classList.remove('active');
      step.classList.add('done');
    }
  }

  // ==========================================================================
  // Error Display
  // ==========================================================================
  function showError(message, title = 'Couldn’t Read Poster') {
    stopBusyProgress();
    const titleEl = $('err-title');
    const msgEl = $('err-msg');
    if (titleEl) titleEl.textContent = title;
    if (msgEl) msgEl.textContent = message;
    setUiState('error');
    if (titleEl) titleEl.focus();
  }

  // ==========================================================================
  // Toast Notifications
  // ==========================================================================
  function showToast(message, type = 'info') {
    const pill = $('toast-pill');
    const textEl = $('toast-text');
    if (!pill || !textEl) return;

    clearTimeout(toastTimer);
    textEl.textContent = message;
    pill.className = 'toast-pill show';

    if (type === 'success') pill.classList.add('toast-success');
    if (type === 'error') pill.classList.add('toast-error');

    toastTimer = setTimeout(() => {
      pill.classList.remove('show');
    }, 2400);
  }

  // ==========================================================================
  // Sample Posters Gallery
  // ==========================================================================
  function cleanFileName(filename) {
    return filename
      .replace(/\.[^.]+$/, '')
      .replace(/[_]+/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  }

  async function loadSamples() {
    const countEl = $('samples-count');
    const gridEl = $('samples-grid');

    try {
      const resp = await fetch('/api/samples');
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      allSamples = data.samples || [];

      if (countEl) countEl.textContent = `${allSamples.length} posters`;
      renderSamplesList('');
    } catch (err) {
      if (countEl) countEl.textContent = 'Offline';
      if (gridEl) {
        gridEl.innerHTML = `
          <div style="grid-column: 1/-1; padding: 24px 12px; text-align: center; color: var(--ink-muted);">
            <p style="margin-bottom: 8px;">Couldn’t load samples from server.</p>
            <button type="button" class="btn-retry" id="btn-samples-retry" style="font-size: 12px; padding: 6px 14px;">Retry</button>
          </div>
        `;
        const retryBtn = $('btn-samples-retry');
        if (retryBtn) retryBtn.onclick = loadSamples;
      }
    }
  }

  function renderSamplesList(query = '') {
    const gridEl = $('samples-grid');
    const clearBtn = $('samples-filter-clear');
    if (!gridEl) return;

    if (clearBtn) clearBtn.hidden = !query;

    const lowerQ = query.trim().toLowerCase();
    const filtered = allSamples.filter((s) => {
      const cleaned = cleanFileName(s.filename).toLowerCase();
      return cleaned.includes(lowerQ) || s.filename.toLowerCase().includes(lowerQ);
    });

    if (filtered.length === 0) {
      gridEl.innerHTML = `
        <div style="grid-column: 1/-1; padding: 32px 12px; text-align: center; color: var(--ink-muted); font-size: 13px;">
          No poster matches "<strong>${escapeHtml(query)}</strong>"
        </div>
      `;
      return;
    }

    gridEl.innerHTML = '';

    // Create IntersectionObserver for lazy thumbnail loading
    const observer = new IntersectionObserver(
      (entries, obs) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const tile = entry.target;
            obs.unobserve(tile);
            loadThumbnail(tile);
          }
        });
      },
      { root: gridEl, rootMargin: '80px' }
    );

    filtered.forEach((sample) => {
      const tile = document.createElement('div');
      tile.className = 'sample-tile';
      tile.dataset.filename = sample.filename;
      tile.tabIndex = 0;
      tile.role = 'option';
      tile.setAttribute('aria-label', `${cleanFileName(sample.filename)} (${sample.type})`);

      if (activeJob && activeJob.sample === sample.filename) {
        tile.classList.add('selected');
        tile.setAttribute('aria-selected', 'true');
      }

      tile.innerHTML = `
        <div class="sample-thumb-wrap">
          <img class="sample-thumb-img" alt="" data-src="/api/preview/${encodeURIComponent(sample.filename)}">
          <div class="sample-thumb-fallback">
            <span class="file-icon">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
              </svg>
            </span>
            <span style="font-size: 10px; font-weight: 700; text-transform: uppercase;">${escapeHtml(sample.type || 'PDF')}</span>
          </div>
          <span class="sample-type-tag">${escapeHtml(sample.type || 'PDF')}</span>
          <span class="sample-selected-check">✓</span>
        </div>
        <div class="sample-name" title="${escapeHtml(sample.filename)}">${escapeHtml(cleanFileName(sample.filename))}</div>
      `;

      tile.onclick = () => selectSample(sample.filename);
      tile.onkeydown = (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          selectSample(sample.filename);
        }
      };

      gridEl.appendChild(tile);
      observer.observe(tile);
    });
  }

  function loadThumbnail(tile) {
    const img = tile.querySelector('.sample-thumb-img');
    const fallback = tile.querySelector('.sample-thumb-fallback');
    if (!img) return;

    const src = img.dataset.src;
    if (!src) return;

    thumbQueue.add(() => {
      return new Promise((resolve) => {
        const testImg = new Image();
        testImg.onload = () => {
          img.src = src;
          img.classList.add('loaded');
          if (fallback) fallback.style.display = 'none';
          resolve();
        };
        testImg.onerror = () => {
          // Graceful fallback tile remains visible
          resolve();
        };
        testImg.src = src;
      });
    });
  }

  function selectSample(filename) {
    $$('.sample-tile').forEach((tile) => {
      const match = tile.dataset.filename === filename;
      tile.classList.toggle('selected', match);
      tile.setAttribute('aria-selected', String(match));
    });

    runAutofill({ sample: filename });
  }

  // ==========================================================================
  // File Validation & Drop Zone Handling
  // ==========================================================================
  function validateFile(file) {
    if (!file) return 'No file chosen.';
    if (!/\.(pdf|png|jpe?g|webp)$/i.test(file.name)) {
      return 'Please choose a valid poster (PDF, PNG, JPG, or WebP).';
    }
    if (file.size === 0) {
      return 'The chosen file is empty (0 bytes).';
    }
    if (file.size > CONFIG.maxMB * 1024 * 1024) {
      const mb = (file.size / (1024 * 1024)).toFixed(1);
      return `File size is ${mb} MB. The maximum allowed limit is ${CONFIG.maxMB} MB.`;
    }
    return null;
  }

  function handleFileSelected(file) {
    const err = validateFile(file);
    if (err) {
      showError(err, 'Invalid Poster File');
      return;
    }

    // Clear active sample selection
    $$('.sample-tile').forEach((t) => {
      t.classList.remove('selected');
      t.removeAttribute('aria-selected');
    });

    runAutofill({ file });
  }

  // ==========================================================================
  // Main API Call: POST /api/autofill
  // ==========================================================================
  async function runAutofill(job) {
    if (activeAbortCtrl) {
      activeAbortCtrl.abort();
    }

    const abortCtrl = new AbortController();
    activeAbortCtrl = abortCtrl;
    activeJob = job;

    setUiState('busy');

    const timeoutHandle = setTimeout(() => {
      abortCtrl.abort('timeout');
    }, CONFIG.timeoutMs);

    try {
      const formData = new FormData();
      if (job.file) {
        formData.append('file', job.file, job.file.name);
      } else {
        formData.append('sample', job.sample);
      }

      const response = await fetch('/api/autofill', {
        method: 'POST',
        body: formData,
        signal: abortCtrl.signal,
      });

      const textPayload = await response.text();
      let data = null;
      try {
        data = JSON.parse(textPayload);
      } catch (e) {
        throw new Error('Server returned an unexpected non-JSON response.');
      }

      if (!response.ok || data.status === 'error') {
        const msg = data.message || `Server responded with status ${response.status}.`;
        throw new Error(msg);
      }

      if (abortCtrl !== activeAbortCtrl) return;

      stopBusyProgress();
      populateResult(data, job);
    } catch (err) {
      if (abortCtrl !== activeAbortCtrl) return;

      if (err.name === 'AbortError' || err === 'timeout') {
        showError('Extraction took too long (timed out). Try a smaller or clearer PDF.', 'Request Timed Out');
      } else if (err instanceof TypeError && err.message.includes('fetch')) {
        showError('Cannot connect to the server at http://localhost:5000. Ensure the app is running.', 'Connection Problem');
      } else {
        showError(err.message || 'An unexpected error occurred during extraction.', 'Extraction Failed');
      }
    } finally {
      clearTimeout(timeoutHandle);
    }
  }

  // ==========================================================================
  // Result Form & Card Population
  // ==========================================================================
  function populateResult(apiData, job) {
    const rawData = apiData.structured_data || {};
    originalData = {};

    SCHEMA_FIELDS.forEach((f) => {
      originalData[f] = rawData[f] != null ? String(rawData[f]) : '';
    });

    // Populate Headers & Meta
    const resultTitleEl = $('result-title');
    const filenameEl = $('result-file-text');
    const statFilledEl = $('stat-filled');
    const statTimeEl = $('stat-time');

    const displayName = apiData.filename || (job.file ? job.file.name : job.sample);
    if (filenameEl) filenameEl.textContent = displayName;

    const execSeconds = apiData.execution_time_seconds != null ? Number(apiData.execution_time_seconds).toFixed(2) : null;
    if (statTimeEl) {
      statTimeEl.textContent = execSeconds ? `⏱ ${execSeconds}s` : '⏱ Instant';
    }

    // Set Poster Preview URL
    setupPosterPreview(job, apiData);

    // Populate Raw Text
    const rawPre = $('raw-ocr-text');
    if (rawPre) {
      rawPre.textContent = apiData.ocr_text || 'No text extracted from document.';
    }

    // Render Form Fields with Staggered Cascading Animation
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let filledCount = 0;

    SCHEMA_FIELDS.forEach((f, idx) => {
      const val = originalData[f] || '';
      if (val.trim()) filledCount++;

      const wrap = document.querySelector(`.field-wrap[data-field="${f}"]`);
      const input = $(`f-${f}`);

      if (input) {
        input.value = val;
      }

      if (f === 'event_type') {
        renderEventTypeChips(val);
      }

      if (wrap) {
        wrap.classList.remove('edited');
        wrap.classList.toggle('missing', !val.trim());

        if (!prefersReducedMotion && val.trim()) {
          setTimeout(() => {
            wrap.classList.add('just-filled');
            setTimeout(() => wrap.classList.remove('just-filled'), 800);
          }, idx * CONFIG.staggerMs);
        }
      }
    });

    if (statFilledEl) {
      statFilledEl.textContent = `${filledCount} of 10 filled`;
      statFilledEl.className = filledCount >= 7 ? 'stat-pill filled-stat' : 'stat-pill time-stat';
    }

    if (resultTitleEl) {
      resultTitleEl.textContent = filledCount === 0 ? 'No Fields Extracted' : 'Review & Refine';
    }

    // Update Live FestNest Event Card
    updateLiveCardPreview();

    // Show Left Viewer & Right Result Form
    const viewerCard = $('viewer-card');
    if (viewerCard) viewerCard.hidden = false;
    switchViewerTab('poster');

    setUiState('result');

    // Announce to Screen Readers
    showToast(`${filledCount} of 10 fields extracted.`, 'success');
  }

  // ==========================================================================
  // Segmented Event Type Chips Control
  // ==========================================================================
  function renderEventTypeChips(selectedValue = '') {
    const container = $('event-type-chips');
    const hiddenInput = $('f-event_type');
    if (!container) return;

    container.innerHTML = '';
    const normSelected = (selectedValue || '').trim();

    // Collect all chips (standard + unrecognised model value if present)
    const chips = [...STANDARD_CATEGORIES];
    let isUnrecognised = false;

    if (normSelected && !STANDARD_CATEGORIES.some((c) => c.toLowerCase() === normSelected.toLowerCase())) {
      chips.push(normSelected);
      isUnrecognised = true;
    }

    chips.forEach((cat) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      const isMatch = normSelected.toLowerCase() === cat.toLowerCase();

      btn.className = 'type-chip-btn' + (isMatch ? ' active' : '');
      if (isUnrecognised && cat === normSelected) {
        btn.classList.add('unrecognised');
        btn.textContent = `${cat} (unrecognised)`;
      } else {
        btn.textContent = cat;
      }

      btn.onclick = () => {
        container.querySelectorAll('.type-chip-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        if (hiddenInput) {
          hiddenInput.value = cat;
          hiddenInput.dispatchEvent(new Event('input', { bubbles: true }));
        }
      };

      container.appendChild(btn);
    });

    if (hiddenInput) {
      hiddenInput.value = normSelected;
    }
  }

  // ==========================================================================
  // Poster Preview & Lightbox
  // ==========================================================================
  function setupPosterPreview(job, apiData) {
    const posterImg = $('poster-preview-img');
    const emptyState = $('poster-empty-state');
    const bannerImg = $('card-preview-banner');
    const lightboxImg = $('lightbox-img');
    const lightboxName = $('lightbox-filename');

    if (currentPosterBlobUrl) {
      URL.revokeObjectURL(currentPosterBlobUrl);
      currentPosterBlobUrl = null;
    }

    let targetSrc = '';
    const filename = apiData?.filename || (job.file ? job.file.name : job.sample);

    if (job.file && /^image\//.test(job.file.type)) {
      targetSrc = currentPosterBlobUrl = URL.createObjectURL(job.file);
    } else if (apiData && apiData.preview_url) {
      targetSrc = apiData.preview_url;
    } else if (job.sample) {
      targetSrc = `/api/preview/${encodeURIComponent(job.sample)}`;
    }

    currentPreviewUrl = targetSrc;

    if (lightboxName) lightboxName.textContent = filename || 'Poster Preview';

    if (targetSrc) {
      if (posterImg) {
        posterImg.hidden = false;
        posterImg.src = targetSrc;
      }
      if (bannerImg) {
        bannerImg.hidden = false;
        bannerImg.src = targetSrc;
      }
      if (lightboxImg) lightboxImg.src = targetSrc;
      if (emptyState) emptyState.hidden = true;
    } else {
      if (posterImg) posterImg.hidden = true;
      if (bannerImg) bannerImg.hidden = true;
      if (emptyState) emptyState.hidden = false;
    }
  }

  function openLightbox() {
    if (!currentPreviewUrl) return;
    const modal = $('lightbox-modal');
    if (modal) {
      modal.hidden = false;
      modal.focus();
    }
  }

  function closeLightbox() {
    const modal = $('lightbox-modal');
    if (modal) modal.hidden = true;
  }

  // ==========================================================================
  // Viewer Tabs (Poster / Raw Text / Live Event Card)
  // ==========================================================================
  function switchViewerTab(tabName) {
    const tabs = ['poster', 'raw', 'card'];
    tabs.forEach((t) => {
      const btn = $(`tab-${t}`);
      const panel = $(`panel-${t}`);
      const isMatch = t === tabName;

      if (btn) btn.setAttribute('aria-selected', String(isMatch));
      if (panel) panel.hidden = !isMatch;
    });
  }

  // ==========================================================================
  // Live FestNest Event Card Sync
  // ==========================================================================
  function updateLiveCardPreview() {
    const getVal = (id) => ($(`f-${id}`) ? $(`f-${id}`).value.trim() : '');

    const name = getVal('event_name') || 'Event Title';
    const type = getVal('event_type') || 'HACKATHON';
    const date = getVal('date') || 'Date not specified';
    const venue = getVal('venue') || 'Venue not specified';
    const organizer = getVal('organizer') || 'Host / College';
    const prize = getVal('prize_pool');

    const titleEl = $('card-preview-title');
    const badgeEl = $('card-preview-badge');
    const dateEl = $('card-preview-date');
    const venueEl = $('card-preview-venue');
    const orgEl = $('card-preview-organizer');
    const prizeEl = $('card-preview-prize');

    if (titleEl) titleEl.textContent = name;
    if (badgeEl) badgeEl.textContent = type.toUpperCase();
    if (dateEl) dateEl.textContent = date;
    if (venueEl) venueEl.textContent = venue;
    if (orgEl) orgEl.textContent = organizer;

    if (prizeEl) {
      if (prize) {
        prizeEl.hidden = false;
        prizeEl.textContent = prize.startsWith('₹') ? prize : `₹${prize}`;
      } else {
        prizeEl.hidden = true;
      }
    }
  }

  // ==========================================================================
  // Form Actions: Copy JSON, Download, Check Form, Revert
  // ==========================================================================
  function readFormData() {
    const result = {};
    SCHEMA_FIELDS.forEach((f) => {
      const input = $(`f-${f}`);
      result[f] = input ? input.value.trim() : '';
    });
    return result;
  }

  async function copyJsonToClipboard() {
    const data = readFormData();
    const jsonStr = JSON.stringify(data, null, 2);

    try {
      await navigator.clipboard.writeText(jsonStr);
      showToast('Copied JSON to clipboard', 'success');

      const copyBtn = $('btn-copy-json');
      if (copyBtn) {
        const origSpan = copyBtn.querySelector('span');
        if (origSpan) {
          const prevText = origSpan.textContent;
          origSpan.textContent = '✓ Copied!';
          setTimeout(() => {
            origSpan.textContent = prevText;
          }, 1500);
        }
      }
    } catch (err) {
      showToast('Clipboard access denied. Please copy manually.', 'error');
    }
  }

  function downloadJsonFile() {
    const data = readFormData();
    const jsonStr = JSON.stringify(data, null, 2);
    const safeName = (data.event_name || 'festnest-event')
      .replace(/[^\w\d-_]+/g, '-')
      .replace(/^-+|-+$/g, '')
      .toLowerCase();

    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${safeName || 'event'}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showToast('Downloaded JSON file', 'success');
  }

  function checkFormRequirements() {
    const data = readFormData();
    const required = [
      { key: 'event_name', label: 'Event Name' },
      { key: 'event_type', label: 'Category' },
      { key: 'date', label: 'Date' },
      { key: 'venue', label: 'Venue' },
    ];

    const missing = required.filter((r) => !data[r.key]);

    if (missing.length === 0) {
      showToast('All essential event details are complete!', 'success');
    } else {
      const names = missing.map((m) => m.label).join(', ');
      showToast(`Please check: ${names}`, 'error');

      // Focus first missing field
      const firstMissing = $(`f-${missing[0].key}`);
      if (firstMissing) {
        firstMissing.focus();
        firstMissing.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }
  }

  function revertAllFields() {
    SCHEMA_FIELDS.forEach((f) => {
      const input = $(`f-${f}`);
      if (input) {
        input.value = originalData[f] || '';
        input.dispatchEvent(new Event('input', { bubbles: true }));
      }
    });
    showToast('Reverted all fields to extracted values', 'info');
  }

  function startOver() {
    setUiState('idle');
    activeJob = null;
    originalData = {};
    const viewerCard = $('viewer-card');
    if (viewerCard) viewerCard.hidden = true;

    $$('.sample-tile').forEach((t) => {
      t.classList.remove('selected');
      t.removeAttribute('aria-selected');
    });

    const fileInput = $('file-input');
    if (fileInput) fileInput.value = '';
  }

  // ==========================================================================
  // Inline Field Copy / Revert
  // ==========================================================================
  function handleInlineFieldActions(e) {
    const copyBtn = e.target.closest('.btn-copy-field');
    const revertBtn = e.target.closest('.btn-revert-field');

    if (copyBtn) {
      const targetId = copyBtn.dataset.for;
      const input = $(targetId);
      if (input) {
        navigator.clipboard.writeText(input.value.trim());
        showToast('Copied field value', 'info');
      }
    }

    if (revertBtn) {
      const targetId = revertBtn.dataset.for;
      const input = $(targetId);
      const fieldName = targetId.replace('f-', '');
      if (input && originalData[fieldName] !== undefined) {
        input.value = originalData[fieldName];
        input.dispatchEvent(new Event('input', { bubbles: true }));
        showToast('Reverted field', 'info');
      }
    }
  }

  // ==========================================================================
  // Raw OCR Text Search Filter
  // ==========================================================================
  function filterRawText(query) {
    const rawPre = $('raw-ocr-text');
    if (!rawPre) return;

    const fullText = rawPre.innerText;
    if (!query.trim()) {
      rawPre.textContent = fullText;
      return;
    }

    const escaped = escapeRegex(query.trim());
    const regex = new RegExp(`(${escaped})`, 'gi');
    rawPre.innerHTML = escapeHtml(fullText).replace(regex, '<mark>$1</mark>');
  }

  // ==========================================================================
  // Global Event Listeners & Shortcuts
  // ==========================================================================
  function setupEventListeners() {
    // Dropzone
    const dropzone = $('dropzone');
    const fileInput = $('file-input');

    if (dropzone && fileInput) {
      dropzone.onclick = () => fileInput.click();
      dropzone.onkeydown = (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          fileInput.click();
        }
      };

      fileInput.onchange = () => {
        if (fileInput.files[0]) {
          handleFileSelected(fileInput.files[0]);
        }
      };

      dropzone.ondragover = (e) => {
        e.preventDefault();
        dropzone.classList.add('drag-over');
      };

      dropzone.ondragleave = () => {
        dropzone.classList.remove('drag-over');
      };

      dropzone.ondrop = (e) => {
        e.preventDefault();
        dropzone.classList.remove('drag-over');
        if (e.dataTransfer.files[0]) {
          handleFileSelected(e.dataTransfer.files[0]);
        }
      };
    }

    // Full-page Drag & Drop Overlay
    const globalOverlay = $('global-drag-overlay');
    let dragCounter = 0;

    window.addEventListener('dragenter', (e) => {
      if (e.dataTransfer && e.dataTransfer.types.includes('Files')) {
        dragCounter++;
        if (globalOverlay) globalOverlay.classList.add('active');
      }
    });

    window.addEventListener('dragleave', (e) => {
      dragCounter--;
      if (dragCounter <= 0 && globalOverlay) {
        dragCounter = 0;
        globalOverlay.classList.remove('active');
      }
    });

    window.addEventListener('dragover', (e) => e.preventDefault());

    window.addEventListener('drop', (e) => {
      e.preventDefault();
      dragCounter = 0;
      if (globalOverlay) globalOverlay.classList.remove('active');
      if (e.dataTransfer && e.dataTransfer.files[0]) {
        handleFileSelected(e.dataTransfer.files[0]);
      }
    });

    // Clipboard Paste Listener (Ctrl+V)
    window.addEventListener('paste', (e) => {
      if (['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
        return; // Don't intercept normal field pastes
      }

      const items = e.clipboardData?.items;
      if (!items) return;

      for (let i = 0; i < items.length; i++) {
        if (items[i].type.startsWith('image/')) {
          const blob = items[i].getAsFile();
          if (blob) {
            showToast('Pasted image from clipboard', 'info');
            handleFileSelected(blob);
            break;
          }
        }
      }
    });

    // Samples Filter Input
    const filterInput = $('samples-filter-input');
    const filterClear = $('samples-filter-clear');
    if (filterInput) {
      filterInput.oninput = (e) => renderSamplesList(e.target.value);
    }
    if (filterClear) {
      filterClear.onclick = () => {
        if (filterInput) {
          filterInput.value = '';
          renderSamplesList('');
          filterInput.focus();
        }
      };
    }

    // Viewer Tab Switching
    ['poster', 'raw', 'card'].forEach((tabName) => {
      const btn = $(`tab-${tabName}`);
      if (btn) btn.onclick = () => switchViewerTab(tabName);
    });

    // Poster Preview Lightbox
    const posterImg = $('poster-preview-img');
    const lightboxClose = $('lightbox-close');
    const lightboxModal = $('lightbox-modal');

    if (posterImg) posterImg.onclick = openLightbox;
    if (lightboxClose) lightboxClose.onclick = closeLightbox;
    if (lightboxModal) {
      lightboxModal.onclick = (e) => {
        if (e.target === lightboxModal) closeLightbox();
      };
    }

    // Raw OCR Text Actions
    const rawSearch = $('raw-search-input');
    const btnCopyRaw = $('btn-copy-raw');
    if (rawSearch) {
      rawSearch.oninput = (e) => filterRawText(e.target.value);
    }
    if (btnCopyRaw) {
      btnCopyRaw.onclick = () => {
        const rawPre = $('raw-ocr-text');
        if (rawPre) {
          navigator.clipboard.writeText(rawPre.innerText);
          showToast('Copied raw extracted text', 'success');
        }
      };
    }

    // Form Field Edit Tracking & Live Card Sync
    const form = $('autofill-form');
    if (form) {
      form.addEventListener('input', (e) => {
        const target = e.target;
        const fieldWrap = target.closest('.field-wrap');
        const fieldName = fieldWrap?.dataset.field;

        if (fieldWrap && fieldName) {
          const currentVal = target.value.trim();
          const origVal = (originalData[fieldName] || '').trim();

          fieldWrap.classList.toggle('edited', currentVal !== origVal);
          fieldWrap.classList.toggle('missing', !currentVal);
        }

        updateLiveCardPreview();
      });

      form.addEventListener('click', handleInlineFieldActions);
      form.addEventListener('submit', (e) => e.preventDefault());
    }

    // Sticky Action Bar Buttons
    const btnCopyJson = $('btn-copy-json');
    const btnDownloadJson = $('btn-download-json');
    const btnValidate = $('btn-validate');
    const btnRevertAll = $('btn-revert-all');
    const btnStartOver = $('btn-start-over');

    if (btnCopyJson) btnCopyJson.onclick = copyJsonToClipboard;
    if (btnDownloadJson) btnDownloadJson.onclick = downloadJsonFile;
    if (btnValidate) btnValidate.onclick = checkFormRequirements;
    if (btnRevertAll) btnRevertAll.onclick = revertAllFields;
    if (btnStartOver) btnStartOver.onclick = startOver;

    // Error & Busy Controls
    const btnCancel = $('btn-cancel');
    const btnRetry = $('btn-retry');
    const btnErrReset = $('btn-err-reset');

    if (btnCancel) {
      btnCancel.onclick = () => {
        if (activeAbortCtrl) activeAbortCtrl.abort();
        stopBusyProgress();
        setUiState('idle');
        showToast('Extraction cancelled', 'info');
      };
    }

    if (btnRetry) {
      btnRetry.onclick = () => {
        if (activeJob) runAutofill(activeJob);
        else setUiState('idle');
      };
    }

    if (btnErrReset) {
      btnErrReset.onclick = startOver;
    }

    // Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
      // Esc closes lightbox or cancels busy
      if (e.key === 'Escape') {
        const modal = $('lightbox-modal');
        if (modal && !modal.hidden) {
          closeLightbox();
          return;
        }
        const busyCard = $('state-busy');
        if (busyCard && !busyCard.hidden && btnCancel) {
          btnCancel.click();
        }
      }

      // / focuses sample search (or global search)
      if (e.key === '/' && !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
        e.preventDefault();
        const searchInput = $('samples-filter-input') || $('global-search-input');
        if (searchInput) searchInput.focus();
      }

      // Ctrl+Enter or Cmd+Enter copies JSON
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const resultCard = $('state-result');
        if (resultCard && !resultCard.hidden && btnCopyJson) {
          e.preventDefault();
          btnCopyJson.click();
        }
      }
    });
  }

  // ==========================================================================
  // Utilities
  // ==========================================================================
  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  // ==========================================================================
  // Initialization
  // ==========================================================================
  async function init() {
    setupEventListeners();

    // Check URL parameters immediately (e.g. ?sample=..., ?state=busy, ?state=error)
    const urlParams = new URLSearchParams(window.location.search);
    const sampleParam = urlParams.get('sample');
    const stateParam = urlParams.get('state');

    if (stateParam === 'busy') {
      setUiState('busy');
    } else if (stateParam === 'error') {
      showError('OCR engine could not detect readable text in the document stream.', 'Extraction Failed');
    } else if (urlParams.get('test_result') === '1') {
      populateResult({
        status: 'success',
        filename: 'IMPERIUM 2.0 - Brochure_.pdf',
        preview_url: '/api/preview/IMPERIUM%202.0%20-%20Brochure_.pdf',
        structured_data: {
          event_name: 'IMPERIUM 2.0',
          event_type: 'Competition',
          date: '16th - 22nd Oct 2026',
          time: '9:00 AM – 5:00 PM',
          venue: 'On-Campus & Virtual',
          organizer: 'Dept. of Professional Accounting and Finance',
          prize_pool: '₹15,000',
          registration_deadline: '16th Oct 2026',
          contact: '+91 99997 64768',
          description: 'National Research Paper Presentation Competition focusing on rethinking architecture and claiming national distinction.'
        },
        execution_time_seconds: 6.12,
        ocr_text: 'NATIONAL RESEARCH PAPER PRESENTATION COMPETITION\nORGANISES\n16th OCT - 22nd OCT\nREGISTRATION DEADLINE\nON-CAMPUS & VIRTUAL\nCASH PRIZES 15,000 PRIZE POOL\nIMPERIUM 2.0'
      }, { sample: 'IMPERIUM 2.0 - Brochure_.pdf' });
    }

    await loadSamples();

    if (sampleParam) {
      selectSample(sampleParam);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
