/* CaptionForge local interface. No framework, no network beyond this machine. */

(() => {
  "use strict";

  const POLL_INTERVAL_MS = 700;
  const SAVE_DELAY_MS = 600;

  // Sizes are the on-disk faster-whisper downloads, rounded.
  const WHISPER_MODELS = [
    { value: "tiny", label: "Tiny", note: "~75 MB · fastest, roughest" },
    { value: "base", label: "Base", note: "~145 MB" },
    { value: "small", label: "Small", note: "~465 MB · a good default" },
    { value: "medium", label: "Medium", note: "~1.5 GB" },
    { value: "large-v3", label: "Large v3", note: "~3 GB · slowest, most accurate" },
  ];
  const TERMINAL = new Set(["completed", "failed", "cancelled"]);

  const el = (id) => document.getElementById(id);
  const ui = {
    meta: el("meta"), form: el("source-form"), url: el("url"),
    inspect: el("inspect-btn"), hint: el("hint"),
    alert: el("alert"), alertTitle: el("alert-title"), alertBody: el("alert-body"),
    video: el("video"), thumb: el("thumb"), title: el("title"), byline: el("byline"),
    tracks: el("tracks"), trackList: el("track-list"), selection: el("selection"),
    controls: el("controls"), language: el("language"), formats: el("formats"),
    run: el("run"), cancel: el("cancel"),
    progress: el("progress"), stage: el("stage"), pct: el("pct"), fill: el("fill"),
    results: el("results"), resultsLabel: el("results-label"),
    files: el("files"), resultsNote: el("results-note"),
    force: el("force"), allowTranslated: el("allow-translated"),
    timestamped: el("timestamped-txt"), overwrite: el("overwrite"),
    postprocess: el("postprocess"), keepAudio: el("keep-audio"),
    models: el("models"), modelCustom: el("model-custom"),
    deviceCuda: el("device-cuda"), cudaNote: el("cuda-note"),
    prompt: el("prompt"), optionsPanel: el("options-panel"),
  };

  const state = {
    token: "",
    defaults: null,
    inspection: null,
    chosenFormats: new Set(),
    jobId: null,
    timer: null,
    preferences: {},
    saveTimer: null,
  };

  /* ---------- token ---------- */

  function claimToken() {
    const params = new URLSearchParams(location.search);
    const token = params.get("t");
    if (!token) return false;
    state.token = token;
    // Keep the token out of the address bar so it cannot leak onward.
    history.replaceState(null, "", location.pathname);
    return true;
  }

  /* ---------- transport ---------- */

  async function api(path, options = {}) {
    const response = await fetch(path, {
      ...options,
      headers: {
        "X-CaptionForge-Token": state.token,
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...(options.headers || {}),
      },
    });
    let payload = null;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }
    if (!response.ok) {
      const message = payload && payload.error
        ? payload.error
        : detailOf(payload) || "CaptionForge could not complete the request.";
      throw new Error(message);
    }
    return payload;
  }

  function detailOf(payload) {
    // FastAPI reports schema problems as a list of {msg, loc} entries.
    if (!payload || !Array.isArray(payload.detail)) return null;
    return payload.detail.map((item) => item.msg).join(" ");
  }

  /* ---------- view helpers ---------- */

  function showAlert(title, message) {
    ui.alertTitle.textContent = title;
    ui.alertBody.textContent = message;
    ui.alert.hidden = false;
  }

  function clearAlert() {
    ui.alert.hidden = true;
  }

  function busy(isBusy) {
    ui.inspect.disabled = isBusy;
    ui.run.disabled = isBusy;
  }

  function formatDuration(seconds) {
    if (seconds === null || seconds === undefined) return "";
    const total = Math.round(seconds);
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const rest = total % 60;
    const pad = (value) => String(value).padStart(2, "0");
    return hours ? `${hours}:${pad(minutes)}:${pad(rest)}` : `${minutes}:${pad(rest)}`;
  }

  function formatSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  /* ---------- startup ---------- */

  async function start() {
    if (!claimToken()) {
      ui.form.hidden = true;
      ui.hint.hidden = true;
      showAlert(
        "Access token missing",
        "Open the link CaptionForge printed in your terminal. It carries the " +
        "token this page needs to talk to the local server."
      );
      return;
    }
    try {
      state.defaults = await api("/api/health");
    } catch (error) {
      showAlert("Cannot reach CaptionForge", error.message);
      return;
    }
    ui.meta.textContent = `v${state.defaults.version} · ${state.defaults.output_directory}`;
    state.preferences = state.defaults.preferences || {};
    applyPreferences();
    watchForChanges();
  }

  function applyPreferences() {
    const saved = state.preferences;
    ui.language.value = saved.language || state.defaults.default_language;
    ui.prompt.value = saved.prompt || "";
    for (const [box, key] of [
      [ui.force, "force"],
      [ui.overwrite, "overwrite"],
      [ui.keepAudio, "keep_audio"],
      [ui.timestamped, "timestamped_txt"],
      [ui.allowTranslated, "allow_translated"],
    ]) {
      box.checked = Boolean(saved[key]);
    }
    // Cleaning subtitles is on unless it was explicitly turned off.
    ui.postprocess.checked = saved.postprocess !== false;
    state.chosenFormats = new Set(initialFormats());
    renderFormats();
    renderModels();
    applyDeviceAvailability();
    // Options that differ from the defaults are worth showing on arrival.
    if (ui.force.checked || ui.overwrite.checked || ui.keepAudio.checked ||
        ui.timestamped.checked || ui.allowTranslated.checked ||
        !ui.postprocess.checked || ui.prompt.value) {
      ui.optionsPanel.open = true;
    }
  }

  /* ---------- remembering ---------- */

  function currentPreferences() {
    return {
      language: ui.language.value.trim() || null,
      formats: [...state.chosenFormats],
      model: chosenModel(),
      device: chosenDevice(),
      prompt: ui.prompt.value.trim() || null,
      force: ui.force.checked,
      overwrite: ui.overwrite.checked,
      keep_audio: ui.keepAudio.checked,
      timestamped_txt: ui.timestamped.checked,
      postprocess: ui.postprocess.checked,
      allow_translated: ui.allowTranslated.checked,
    };
  }

  function rememberChoices() {
    clearTimeout(state.saveTimer);
    state.saveTimer = setTimeout(async () => {
      try {
        await api("/api/preferences", {
          method: "PUT",
          body: JSON.stringify(currentPreferences()),
        });
      } catch {
        // Remembering is a convenience; never interrupt the real work for it.
      }
    }, SAVE_DELAY_MS);
  }

  function watchForChanges() {
    ui.controls.addEventListener("change", rememberChoices);
    ui.controls.addEventListener("input", rememberChoices);
  }

  function renderModels() {
    const wanted = state.preferences.model || state.defaults.default_model;
    const known = WHISPER_MODELS.some((entry) => entry.value === wanted);
    const options = [
      ...WHISPER_MODELS,
      { value: "", label: "Something else", note: "A model name or a folder on this computer" },
    ];
    ui.models.replaceChildren();
    for (const option of options) {
      const isCustom = option.value === "";
      const label = document.createElement("label");
      label.className = "radio";
      const input = document.createElement("input");
      input.type = "radio";
      input.name = "model";
      input.value = option.value;
      input.checked = isCustom ? !known : option.value === wanted;
      input.addEventListener("change", () => {
        ui.modelCustom.hidden = !isCustom;
        if (isCustom) ui.modelCustom.focus();
      });
      const text = document.createElement("span");
      const name = document.createElement("b");
      name.textContent = option.label;
      const note = document.createElement("em");
      note.textContent = option.note;
      text.append(name, note);
      label.append(input, text);
      ui.models.append(label);
      if (input.checked && isCustom) {
        ui.modelCustom.hidden = false;
        ui.modelCustom.value = wanted;
      }
    }
  }

  function applyDeviceAvailability() {
    const wanted = state.preferences.device || state.defaults.default_device;
    const preferred = document.querySelector(
      `input[name="device"][value="${wanted}"]`
    );
    if (preferred && !preferred.disabled) preferred.checked = true;
    if (state.defaults.cuda_available) return;
    ui.deviceCuda.disabled = true;
    ui.cudaNote.textContent = cudaReason();
    if (ui.deviceCuda.checked) {
      const fallback = document.querySelector('input[name="device"][value="auto"]');
      if (fallback) fallback.checked = true;
    }
  }

  function cudaReason() {
    const missing = state.defaults.cuda_missing_libraries || [];
    if (state.defaults.cuda_device_present && missing.length) {
      // The card is there; CTranslate2 just cannot load what it links against.
      return `Detected, but ${missing.join(" and ")} will not load — ` +
        "install nvidia-cublas-cu12 and nvidia-cudnn-cu12 to use it";
    }
    return "No NVIDIA graphics card detected on this computer";
  }

  function chosenModel() {
    const picked = document.querySelector('input[name="model"]:checked');
    if (!picked) return null;
    if (picked.value) return picked.value;
    return ui.modelCustom.value.trim() || null;
  }

  function chosenDevice() {
    const picked = document.querySelector('input[name="device"]:checked');
    return picked ? picked.value : null;
  }

  function initialFormats() {
    // new Set(undefined) is silently empty, which would leave every chip
    // unchecked. Fall back so a server older than this script still works.
    const defaults = state.defaults;
    const candidates = [
      (state.preferences || {}).formats,
      defaults.initial_formats,
      (defaults.default_formats || []).slice(0, 1),
      ["srt"],
    ];
    return candidates.find((entry) => Array.isArray(entry) && entry.length);
  }

  function renderFormats() {
    ui.formats.replaceChildren();
    for (const format of state.defaults.supported_formats) {
      const item = document.createElement("li");
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.textContent = format;
      chip.setAttribute("aria-pressed", String(state.chosenFormats.has(format)));
      chip.classList.toggle("picked", state.chosenFormats.has(format));
      chip.addEventListener("click", () => {
        if (state.chosenFormats.has(format)) state.chosenFormats.delete(format);
        else state.chosenFormats.add(format);
        chip.classList.toggle("picked");
        chip.setAttribute("aria-pressed", String(state.chosenFormats.has(format)));
        rememberChoices();
      });
      item.append(chip);
      ui.formats.append(item);
    }
  }

  /* ---------- inspection ---------- */

  ui.form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const url = ui.url.value.trim();
    if (!url) return;
    clearAlert();
    stopPolling();
    ui.results.hidden = true;
    ui.progress.hidden = true;
    busy(true);
    ui.inspect.textContent = "Looking up…";
    try {
      state.inspection = await api("/api/inspect", {
        method: "POST",
        body: JSON.stringify({
          url,
          language: ui.language.value.trim() || null,
          allow_translated: ui.allowTranslated.checked,
        }),
      });
      renderInspection(state.inspection);
    } catch (error) {
      ui.video.hidden = true;
      ui.tracks.hidden = true;
      showAlert("Could not read that video", error.message);
    } finally {
      busy(false);
      ui.inspect.textContent = "Look up";
    }
  });

  function renderInspection(result) {
    const video = result.video;
    ui.title.textContent = video.title;
    const parts = [video.channel_name, formatDuration(video.duration_seconds)];
    ui.byline.textContent = parts.filter(Boolean).join(" · ");
    if (video.thumbnail_url) {
      ui.thumb.src = video.thumbnail_url;
      ui.thumb.hidden = false;
    } else {
      ui.thumb.removeAttribute("src");
      ui.thumb.hidden = true;
    }
    ui.video.hidden = false;

    // YouTube publishes ~155 machine translations of its own transcription.
    // Show the real tracks; keep the translations behind one line.
    const all = [...result.manual_tracks, ...result.automatic_tracks];
    const real = all.filter((track) => !track.is_translated);
    const translated = all.filter((track) => track.is_translated);
    const selected = result.selected_track;

    ui.trackList.replaceChildren();
    for (const track of real) ui.trackList.append(trackChip(track, selected));

    if (!all.length) {
      const empty = document.createElement("li");
      empty.className = "note";
      empty.textContent = "None published for this video.";
      ui.trackList.append(empty);
    }

    if (translated.length) {
      const item = document.createElement("li");
      const more = document.createElement("button");
      more.type = "button";
      more.className = "chip";
      more.textContent = `+${translated.length} machine-translated`;
      more.addEventListener("click", () => {
        item.remove();
        for (const track of translated) {
          ui.trackList.append(trackChip(track, selected));
        }
      }, { once: true });
      item.append(more);
      ui.trackList.append(item);
    }

    ui.selection.textContent = result.selected_track
      ? `Will export the highlighted track (${result.selection_reason || "preferred match"}).`
      : "No matching track, so the audio will be transcribed on this computer.";
    ui.tracks.hidden = false;
    ui.controls.hidden = false;
  }

  function trackChip(track, selected) {
    const item = document.createElement("li");
    const chip = document.createElement("span");
    chip.className = "chip";
    const picked = selected &&
      track.language_code === selected.language_code &&
      track.is_automatic === selected.is_automatic;
    chip.classList.toggle("picked", Boolean(picked));
    const code = document.createElement("span");
    code.textContent = track.language_code;
    const kind = document.createElement("span");
    kind.className = "kind";
    kind.textContent = [
      track.is_automatic ? "auto" : "manual",
      track.is_translated ? "translated" : null,
    ].filter(Boolean).join(" · ");
    chip.append(code, kind);
    item.append(chip);
    return item;
  }

  /* ---------- jobs ---------- */

  ui.run.addEventListener("click", async () => {
    if (!state.chosenFormats.size) {
      showAlert("Pick a format", "Choose at least one output format.");
      return;
    }
    clearAlert();
    ui.results.hidden = true;
    busy(true);
    setProgress("Queued", 0);
    ui.cancel.disabled = false;
    ui.progress.hidden = false;
    try {
      const job = await api("/api/jobs", {
        method: "POST",
        body: JSON.stringify({
          url: ui.url.value.trim(),
          language: ui.language.value.trim() || null,
          formats: [...state.chosenFormats],
          model: chosenModel(),
          device: chosenDevice(),
          prompt: ui.prompt.value.trim() || null,
          force: ui.force.checked,
          overwrite: ui.overwrite.checked,
          keep_audio: ui.keepAudio.checked,
          timestamped_txt: ui.timestamped.checked,
          postprocess: ui.postprocess.checked,
          allow_translated: ui.allowTranslated.checked,
        }),
      });
      state.jobId = job.id;
      rememberChoices();
      state.timer = setInterval(poll, POLL_INTERVAL_MS);
      poll();
    } catch (error) {
      ui.progress.hidden = true;
      busy(false);
      showAlert("Could not start", error.message);
    }
  });

  ui.cancel.addEventListener("click", async () => {
    if (!state.jobId) return;
    ui.cancel.disabled = true;
    ui.stage.textContent = "Cancelling…";
    try {
      await api(`/api/jobs/${state.jobId}/cancel`, { method: "POST" });
    } catch (error) {
      showAlert("Could not cancel", error.message);
    }
  });

  function stopPolling() {
    if (state.timer) clearInterval(state.timer);
    state.timer = null;
  }

  function setProgress(stage, percent) {
    ui.stage.textContent = stage;
    ui.pct.textContent = `${Math.round(percent)}%`;
    ui.fill.style.width = `${percent}%`;
  }

  async function poll() {
    if (!state.jobId) return;
    let job;
    try {
      job = await api(`/api/jobs/${state.jobId}`);
    } catch (error) {
      stopPolling();
      busy(false);
      ui.progress.hidden = true;
      showAlert("Lost track of the job", error.message);
      return;
    }
    setProgress(job.stage, job.percent);
    if (!TERMINAL.has(job.status)) return;

    stopPolling();
    busy(false);
    ui.progress.hidden = true;
    if (job.status === "failed") {
      showAlert("Job failed", job.error || "CaptionForge could not finish.");
      return;
    }
    if (job.status === "cancelled") {
      showAlert("Cancelled", "No incomplete output was kept.");
      return;
    }
    renderResults(job);
  }

  function renderResults(job) {
    ui.resultsLabel.textContent = job.used_existing_captions
      ? "Exported the video's own captions"
      : "Transcribed on this computer";
    ui.files.replaceChildren();
    for (const file of job.files) {
      const item = document.createElement("li");
      const link = document.createElement("a");
      link.href = `/api/jobs/${job.id}/files/${encodeURIComponent(file.name)}?t=${encodeURIComponent(state.token)}`;
      link.setAttribute("download", file.name);
      const name = document.createElement("span");
      name.textContent = file.name;
      const size = document.createElement("span");
      size.className = "size";
      size.textContent = formatSize(file.size_bytes);
      const arrow = document.createElement("span");
      arrow.className = "arrow";
      arrow.textContent = "↓";
      link.append(name, size, arrow);
      item.append(link);
      ui.files.append(item);
    }
    const notes = [`Saved to ${state.defaults.output_directory}`];
    if (job.transcription) {
      const info = job.transcription;
      const probability = info.language_probability !== null
        ? ` (${Math.round(info.language_probability * 100)}% confident)`
        : "";
      notes.push(
        `${info.model_name} on ${info.device}, detected ${info.detected_language}${probability}`
      );
    }
    ui.resultsNote.textContent = notes.join(" · ");
    ui.results.hidden = false;
  }

  start();
})();
