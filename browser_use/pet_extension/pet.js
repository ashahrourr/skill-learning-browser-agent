(async () => {
  const existingPet = document.getElementById("browser-use-pet-agent");
  if (existingPet?.nextElementSibling?.tagName === "SECTION") existingPet.nextElementSibling.remove();
  existingPet?.remove();
  document.getElementById("browser-use-pet-agent-panel")?.remove();
  document.getElementById("browser-use-intent-cursor")?.remove();
  document.getElementById("browser-use-pet-agent-badge")?.remove();

  const siteNative = globalThis.__browserUsePetSiteNative;
  if (siteNative?.ready) {
    const { enabled } = await siteNative.ready;
    if (!enabled) return;
  }

  const color = "#e9894c";
  const outline = "#6f432e";
  const activeStates = new Set(["queued", "running", "waiting", "stopping"]);
  const visibleStates = new Set(["queued", "running", "waiting", "stopping", "completed", "failed", "stopped"]);
  const host = document.createElement("div");
  host.id = "browser-use-pet-agent";
  host.setAttribute("data-browser-use-pet", "true");
  host.dataset.version = "6.6.4";
  host.style.cssText = `
    all: initial; position: fixed; inset: 0; width: 0; height: 0;
    z-index: 2147483647; pointer-events: none;
  `;
  const root = host.attachShadow({ mode: "open" });
  const styles = document.createElement("style");
  styles.textContent = `
    .cat-bubble::after {
      position: absolute;
      width: 18px;
      height: 18px;
      box-sizing: border-box;
      border-right: 3px solid ${outline};
      border-bottom: 3px solid ${outline};
      background: #fff7ea;
      content: "";
    }
    .cat-bubble[data-tail="bottom"]::after {
      bottom: -11px;
      left: var(--tail-offset, 50%);
      transform: translateX(-50%) rotate(45deg);
    }
    .cat-bubble[data-tail="left"]::after {
      left: -11px;
      top: var(--tail-offset, 50%);
      transform: translateY(-50%) rotate(135deg);
    }
    .cat-bubble[data-tail="right"]::after {
      right: -11px;
      top: var(--tail-offset, 50%);
      transform: translateY(-50%) rotate(-45deg);
    }
    .cat-bubble.is-visible {
      opacity: 1;
      pointer-events: auto;
      transform: scale(1);
    }
    .cat-bubble button:hover { filter: brightness(1.06); }
    .cat-bubble button:active { transform: translate(2px, 2px); box-shadow: none !important; }
  `;
  const getMountTarget = () => {
    const topLayerElements = [...document.querySelectorAll("dialog[open]")];
    return topLayerElements[topLayerElements.length - 1] || document.documentElement;
  };

  const ensureMounted = () => {
    const mountTarget = getMountTarget();
    if (host.parentElement !== mountTarget) mountTarget.append(host);
  };

  const send = async (message) => {
    const response = await chrome.runtime.sendMessage(message);
    if (!response?.ok) throw new Error(response?.error || "Website Pet extension request failed");
    return response.data;
  };
  const sessionInfo = await send({ type: "pet:get-session" });
  const petSessionId = sessionInfo?.session_id;
  if (petSessionId) {
    document.documentElement.setAttribute("data-browser-use-pet-session-id", petSessionId);
  }

  const panel = document.createElement("section");
  panel.className = "cat-bubble";
  panel.dataset.tail = "bottom";
  panel.style.cssText = `
    position: fixed; right: 24px; bottom: 188px; width: min(360px, calc(100vw - 32px));
    max-height: calc(100vh - 32px); overflow: visible;
    box-sizing: border-box; padding: 14px; border: 3px solid ${outline};
    border-radius: 18px; background: #fff7ea; box-shadow: 0 10px 24px rgba(79, 50, 40, .2);
    display: flex; flex-direction: column; color: #4f3228;
    font: 600 13px/1.4 system-ui, sans-serif; z-index: 2147483647;
    opacity: 0; pointer-events: none; transform: scale(.92);
    transform-origin: var(--bubble-origin, bottom right);
    transition: left 180ms ease, bottom 180ms ease, opacity 130ms ease, transform 160ms ease;
  `;

  const status = document.createElement("div");
  status.textContent = "Ready";
  status.style.cssText = `
    flex: 1 1 auto; min-height: 0; max-height: calc(100vh - 240px);
    margin-bottom: 10px; overflow-y: auto; color: #4f3228;
    line-height: 1.35; white-space: pre-wrap;
  `;

  const taskForm = document.createElement("form");
  taskForm.style.cssText = "display: flex; gap: 8px;";
  const taskInput = document.createElement("input");
  taskInput.type = "text";
  taskInput.placeholder = "What should I do on this page?";
  taskInput.autocomplete = "off";
  taskInput.style.cssText = `
    box-sizing: border-box; width: 100%; padding: 9px 10px; border: 2px solid ${outline};
    border-radius: 4px; color: #4f3228; background: #fffdf8; outline-color: #5ea9c9;
    font: 600 13px system-ui, sans-serif;
  `;
  const run = document.createElement("button");
  run.type = "submit";
  run.textContent = "Run";

  const replyForm = document.createElement("form");
  replyForm.style.cssText = "display: none; gap: 8px;";
  const replyInput = document.createElement("textarea");
  replyInput.placeholder = "Type your reply";
  replyInput.autocomplete = "off";
  replyInput.rows = 3;
  replyInput.style.cssText = `${taskInput.style.cssText} resize: vertical;`;
  // Keep keystrokes typed into the pet's inputs from leaking to the page (e.g. arrow keys moving a
  // game's tiles). A WINDOW-level CAPTURE listener fires before any page handler — capture OR bubble —
  // and stops propagation while focus is inside the pet. We do NOT preventDefault, so the character is
  // still typed into our field; we only keep the page's own key handlers from seeing it.
  // (host.contains(activeElement) is true whenever focus is inside the pet, incl. its shadow root.)
  const trapPetKeys = (event) => {
    const path = typeof event.composedPath === "function" ? event.composedPath() : [];
    if (path.includes(host) || host.contains(document.activeElement)) event.stopImmediatePropagation();
  };
  for (const type of ["keydown", "keypress", "keyup"]) {
    window.addEventListener(type, trapPetKeys, true);
  }
  const reply = document.createElement("button");
  reply.type = "submit";
  reply.textContent = "Reply";

  const stop = document.createElement("button");
  stop.type = "button";
  stop.textContent = "Stop";
  stop.style.cssText = "display: none; margin-top: 10px; background: #f7d9c3;";

  for (const button of [run, reply, stop]) {
    button.style.cssText += `
      border: 2px solid ${outline}; border-radius: 4px; padding: 7px 12px; color: #4f3228;
      box-shadow: 2px 2px 0 rgba(111, 67, 46, .35);
      cursor: pointer; font: 800 13px system-ui, sans-serif;
    `;
  }
  run.style.background = color;
  reply.style.background = color;

  let lastPetPosition = { x: innerWidth - 112, y: innerHeight - 12 };
  const setPanelVisible = (visible) => {
    panel.classList.toggle("is-visible", visible);
    panel.style.opacity = visible ? "1" : "0";
    panel.style.pointerEvents = visible ? "auto" : "none";
    panel.style.transform = visible ? "scale(1)" : "scale(.92)";
  };
  const isPanelVisible = () => panel.classList.contains("is-visible");

  const positionPanel = ({ x, y }) => {
    lastPetPosition = { x, y };
    const panelWidth = Math.min(360, innerWidth - 32);
    const catHalfWidth = 120;
    const catHeight = 112;
    const gap = 38;
    const panelHeight = Math.min(innerHeight - 32, panel.getBoundingClientRect().height || panel.scrollHeight || 118);
    const clampBottom = (bottom) => Math.max(16, Math.min(innerHeight - panelHeight - 16, bottom));
    const catTop = y - catHeight;
    const placeAbove = catTop - panelHeight - gap >= 16;
    const maxLeft = Math.max(16, innerWidth - panelWidth - 16);
    panel.style.right = "auto";
    if (placeAbove) {
      const panelLeft = Math.max(16, Math.min(maxLeft, x - panelWidth / 2));
      panel.style.left = `${panelLeft}px`;
      panel.style.bottom = `${clampBottom(innerHeight - catTop + gap)}px`;
      panel.dataset.tail = "bottom";
      panel.style.setProperty("--tail-offset", `${Math.max(22, Math.min(panelWidth - 22, x - panelLeft))}px`);
      panel.style.setProperty("--bubble-origin", `${x - panelLeft}px bottom`);
      return;
    }

    const placeBesideRight = x + catHalfWidth + gap + panelWidth <= innerWidth - 16;
    const panelLeft = placeBesideRight ? x + catHalfWidth + gap : Math.max(16, x - catHalfWidth - gap - panelWidth);
    const panelBottom = clampBottom(innerHeight - y - panelHeight / 2);
    const panelTop = innerHeight - panelBottom - panelHeight;
    const tailOffset = y - catHeight / 2 - panelTop;
    panel.style.left = `${panelLeft}px`;
    panel.style.bottom = `${panelBottom}px`;
    panel.dataset.tail = placeBesideRight ? "left" : "right";
    panel.style.setProperty("--tail-offset", `${Math.max(22, Math.min(panelHeight - 22, tailOffset))}px`);
    panel.style.setProperty("--bubble-origin", `${placeBesideRight ? "left" : "right"} center`);
  };

  window.addEventListener("browser-use-pet-position", (event) => positionPanel(event.detail));

  const render = (petStatus) => {
    status.textContent = petStatus.message || "Ready";
    const isWaiting = petStatus.state === "waiting";
    const isWorking = petStatus.state === "queued" || petStatus.state === "running" || petStatus.state === "stopping";
    taskForm.style.display = activeStates.has(petStatus.state) ? "none" : "flex";
    replyForm.style.display = isWaiting ? "flex" : "none";
    stop.style.display = activeStates.has(petStatus.state) ? "block" : "none";
    globalThis.__browserUsePetCompanion?.setMode(isWaiting ? "waiting" : isWorking ? "thinking" : "idle");
    if (visibleStates.has(petStatus.state)) {
      setPanelVisible(true);
      positionPanel(lastPetPosition);
      requestAnimationFrame(() => positionPanel(lastPetPosition));
    }
    if (isWaiting) {
      status.textContent = petStatus.question || petStatus.message;
      replyInput.focus();
    }
  };

  window.addEventListener("browser-use-pet-toggle", async () => {
    const shouldOpen = !isPanelVisible();
    if (shouldOpen) positionPanel(lastPetPosition);
    setPanelVisible(shouldOpen);
    if (isPanelVisible()) {
      await refresh();
      positionPanel(lastPetPosition);
      if (replyForm.style.display === "flex") replyInput.focus();
      else taskInput.focus();
    }
  });

  taskForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const task = taskInput.value.trim();
    if (!task) return;
    const originToken = `${petSessionId || "tab"}:${Date.now()}:${Math.random().toString(36).slice(2)}`;
    document.documentElement.setAttribute("data-browser-use-pet-origin-token", originToken);
    run.disabled = true;
    try {
      await send({ type: "pet:start-task", task, url: window.location.href, origin_token: originToken });
      taskInput.value = "";
      render({ state: "queued", message: "Task queued" });
    } catch (error) {
      status.textContent = "Start the local Website Pet bridge";
      console.error("Website Pet bridge error", error);
    } finally {
      run.disabled = false;
    }
  });

  replyForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const answer = replyInput.value.trim();
    if (!answer) return;
    await send({ type: "pet:reply", reply: answer });
    replyInput.value = "";
    render({ state: "running", message: "Working..." });
  });

  stop.addEventListener("click", async () => {
    await send({ type: "pet:stop" });
    render({ state: "stopping", message: "Stopping task..." });
  });

  const refresh = async () => {
    try {
      ensureMounted();
      render(await send({ type: "pet:get-status" }));
    } catch {
      status.textContent = "Start the local Website Pet bridge";
    }
  };

  taskForm.append(taskInput, run);
  replyForm.append(replyInput, reply);
  panel.append(status, replyForm, taskForm, stop);
  root.append(styles, panel);
  ensureMounted();
  refresh();
  setInterval(refresh, 750);
})();
