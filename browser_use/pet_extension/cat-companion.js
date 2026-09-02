(async () => {
  const siteNative = globalThis.__browserUsePetSiteNative;
  if (siteNative?.ready) {
    const { enabled } = await siteNative.ready;
    if (!enabled) {
      document.getElementById("browser-use-pet-companion")?.remove();
      return;
    }
  }

  const PIXI = globalThis.PIXI;
  if (!PIXI) throw new Error("PixiJS runtime is not loaded");
  PIXI.loadTextures.config.preferWorkers = false;
  PIXI.loadTextures.config.preferCreateImageBitmap = false;

  const FRAME_WIDTH = 80;
  const FRAME_HEIGHT = 64;
  const SCALE = 3;
  const CANVAS_WIDTH = FRAME_WIDTH * SCALE;
  const CANVAS_HEIGHT = FRAME_HEIGHT * SCALE;
  const SPRITE_X = CANVAS_WIDTH / 2;
  const SPRITE_FLOOR_Y = CANVAS_HEIGHT - 8;
  const EDGE_MARGIN = 54;
  const MIN_X = SPRITE_X;
  const MIN_Y = 96;
  const MIN_TRAVEL_MS = 1100;
  const MAX_TRAVEL_MS = 2400;
  const ARRIVAL_ACTION_LEAD_MS = 520;
  const TARGET_LINGER_MS = 2600;
  const MIN_SEGMENT_MS = 260;
  const TARGET_SIDE_OFFSET = 90;
  const TARGET_VERTICAL_OFFSET = 90;
  const assets = {
    idle: { file: "IDLE.png", frames: 8, speed: 0.12 },
    walk: { file: "WALK.png", frames: 12, speed: 0.16 },
    run: { file: "RUN.png", frames: 8, speed: 0.2 },
    jump: { file: "JUMP.png", frames: 3, speed: 0.14, loop: false },
    runningJump: { file: "RUNNING JUMP.png", frames: 3, speed: 0.14 },
    attack: { file: "ATTACK 1.png", frames: 8, speed: 0.2, loop: false },
    hurt: { file: "HURT.png", frames: 4, speed: 0.14, loop: false }
  };

  const assetUrl = (file) => chrome.runtime.getURL(`public/assets/cat/${file}`);
  const makeFrames = async ({ file, frames }) => {
    const sheet = await PIXI.Assets.load(assetUrl(file));
    sheet.source.style.scaleMode = "nearest";
    return Array.from({ length: frames }, (_, index) =>
      new PIXI.Texture({
        source: sheet.source,
        frame: new PIXI.Rectangle(index * FRAME_WIDTH, 0, FRAME_WIDTH, FRAME_HEIGHT)
      })
    );
  };

  document.getElementById("browser-use-pet-companion")?.remove();
  const canvas = document.createElement("canvas");
  canvas.id = "browser-use-pet-companion";
  canvas.width = CANVAS_WIDTH;
  canvas.height = CANVAS_HEIGHT;
  canvas.title = "Open browser agent";
  canvas.style.cssText = `
    position: fixed; left: 0; top: 0; width: ${CANVAS_WIDTH}px; height: ${CANVAS_HEIGHT}px;
    z-index: 2147483647; pointer-events: auto; cursor: pointer;
    image-rendering: pixelated;
  `;
  document.documentElement.append(canvas);
  canvas.addEventListener("click", () => window.dispatchEvent(new CustomEvent("browser-use-pet-toggle")));

  const app = new PIXI.Application();
  const home = () => ({ x: innerWidth - (CANVAS_WIDTH - SPRITE_X), y: innerHeight - 12 });
  const initial = home();
  const state = {
    mode: "idle",
    x: initial.x,
    y: initial.y,
    targetX: initial.x,
    targetY: initial.y,
    temporaryUntil: 0,
    returningToLedge: false,
    lookAtX: null,
    lookAtY: null,
    travel: null,
    travelQueue: [],
    returnTimer: 0
  };
  let sprite;
  let currentAnimation = "";
  const textures = {};

  const clampPosition = ({ x, y }) => ({
    x: Math.max(MIN_X, Math.min(innerWidth - (CANVAS_WIDTH - SPRITE_X), x)),
    y: Math.max(MIN_Y, Math.min(innerHeight - (CANVAS_HEIGHT - SPRITE_FLOOR_Y), y))
  });

  const placeCanvas = () => {
    if (!canvas.isConnected) document.documentElement.append(canvas);
    const visiblePosition = clampPosition(state);
    state.x = visiblePosition.x;
    state.y = visiblePosition.y;
    canvas.style.left = `${state.x - SPRITE_X}px`;
    canvas.style.top = `${state.y - SPRITE_FLOOR_Y}px`;
    window.dispatchEvent(new CustomEvent("browser-use-pet-position", { detail: { x: state.x, y: state.y } }));
  };
  placeCanvas();

  const setAnimation = (name, { restart = false } = {}) => {
    const requested = textures[name] ? name : "idle";
    if (!sprite || (!restart && currentAnimation === requested)) return;
    const config = assets[requested] || assets.idle;
    currentAnimation = requested;
    sprite.textures = textures[requested];
    sprite.animationSpeed = config.speed;
    sprite.loop = config.loop !== false;
    sprite.gotoAndPlay(0);
  };

  const setTemporaryAnimation = (name, duration = 650) => {
    state.mode = name;
    state.temporaryUntil = performance.now() + duration;
    setAnimation(name, { restart: true });
  };

  const faceDirection = (direction) => {
    if (!sprite || direction === 0) return;
    sprite.scale.x = direction > 0 ? -SCALE : SCALE;
  };

  const faceLookTarget = () => {
    if (state.lookAtX === null) return;
    faceDirection(state.lookAtX - state.x);
  };

  const easeInOut = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);

  const chooseMoveAnimation = (fromX, fromY, toX, toY, label) => {
    if (label === "input") return "walk";
    const dx = Math.abs(toX - fromX);
    const dy = Math.abs(toY - fromY);
    const distance = Math.hypot(dx, dy);
    if (dy > 150 && distance > 340) return "runningJump";
    if (distance > 420) return "run";
    return "walk";
  };

  const startTravelSegment = (target, { mode = "walk", duration_ms = 1000, returning = false } = {}) => {
    const distance = Math.hypot(target.x - state.x, target.y - state.y);
    const travelMs = Math.max(MIN_SEGMENT_MS, Math.min(MAX_TRAVEL_MS, duration_ms || distance * 3.2));
    state.targetX = target.x;
    state.targetY = target.y;
    state.mode = mode;
    state.returningToLedge = returning;
    state.travel = {
      startX: state.x,
      startY: state.y,
      endX: target.x,
      endY: target.y,
      startedAt: performance.now(),
      duration: travelMs
    };
    setAnimation(mode);
    return travelMs;
  };

  const startTravel = (target, options = {}) => {
    state.travelQueue = [];
    return startTravelSegment(target, options);
  };

  const startTravelPath = (segments, { returning = false } = {}) => {
    const path = segments.filter(Boolean);
    if (!path.length) return 0;
    state.travelQueue = path.slice(1).map((segment, index) => ({
      ...segment,
      returning: returning && index === path.length - 2
    }));
    startTravelSegment(path[0], {
      mode: path[0].mode,
      duration_ms: path[0].duration_ms,
      returning: returning && path.length === 1
    });
    return path.reduce((total, segment) => total + segment.duration_ms, 0);
  };

  const nextTravelSegment = () => {
    const segment = state.travelQueue.shift();
    if (!segment) return false;
    startTravelSegment(segment, {
      mode: segment.mode,
      duration_ms: segment.duration_ms,
      returning: Boolean(segment.returning)
    });
    return true;
  };

  const planTravelPath = (target, { label = "click", duration_ms = 1200 } = {}) => {
    const totalMs = Math.max(MIN_TRAVEL_MS, Math.min(MAX_TRAVEL_MS, duration_ms));
    const dx = target.x - state.x;
    const dy = target.y - state.y;
    const distance = Math.hypot(dx, dy);
    if (label === "input" || distance < 260 || Math.abs(dy) < 140) {
      return [{ ...target, mode: chooseMoveAnimation(state.x, state.y, target.x, target.y, label), duration_ms: totalMs }];
    }

    const segments = [];
    const horizontalY = state.y;
    const needsHorizontal = Math.abs(dx) > 90;
    const verticalSteps = Math.abs(dy) > 320 ? 2 : 1;
    const horizontalMs = needsHorizontal ? Math.max(320, Math.round(totalMs * 0.36)) : 0;
    const finalMs = Math.max(260, Math.round(totalMs * 0.18));
    const jumpMs = Math.max(320, Math.floor((totalMs - horizontalMs - finalMs) / verticalSteps));

    if (needsHorizontal) {
      segments.push({
        x: target.x,
        y: horizontalY,
        mode: Math.abs(dx) > 420 ? "run" : "walk",
        duration_ms: horizontalMs
      });
    }

    const jumpStartY = needsHorizontal ? horizontalY : state.y;
    for (let index = 1; index <= verticalSteps; index += 1) {
      const progress = index / verticalSteps;
      segments.push({
        x: target.x,
        y: jumpStartY + (target.y - jumpStartY) * progress,
        mode: verticalSteps > 1 || Math.abs(dy) > 190 ? "runningJump" : "jump",
        duration_ms: jumpMs
      });
    }

    segments.push({ ...target, mode: "walk", duration_ms: finalMs });
    return segments;
  };

  const isUsableSurface = (element) => {
    if (!(element instanceof HTMLElement)) return false;
    if (element.id === "browser-use-pet-companion" || element.closest("#browser-use-pet-agent")) return false;
    const rect = element.getBoundingClientRect();
    const style = getComputedStyle(element);
    return (
      style.display !== "none" &&
      style.visibility !== "hidden" &&
      rect.width >= 140 &&
      rect.height >= 80 &&
      rect.width < innerWidth * 0.94 &&
      rect.height < innerHeight * 0.94 &&
      rect.bottom > 0 &&
      rect.right > 0 &&
      rect.top < innerHeight &&
      rect.left < innerWidth
    );
  };

  const parkOnPageSurface = ({ animate = true } = {}) => {
    const position = clampPosition({
      x: animate && state.x < innerWidth / 2 ? EDGE_MARGIN : innerWidth - EDGE_MARGIN,
      y: innerHeight - 8
    });
    if (animate) {
      startTravel(position, { mode: "run", duration_ms: 1150, returning: true });
    } else {
      state.x = position.x;
      state.y = position.y;
      state.targetX = position.x;
      state.targetY = position.y;
      state.travel = null;
      state.mode = "idle";
      state.returningToLedge = false;
      setAnimation("idle");
    }
    placeCanvas();
  };

  const findNearbySurface = (actionX, actionY) => {
    const dialogs = document.querySelectorAll('dialog, [role="dialog"], [aria-modal="true"], [class*="modal"], [class*="dialog"]');
    const nearby = document.elementsFromPoint(actionX, actionY).flatMap((element) => {
      const ancestors = [];
      for (let current = element; current instanceof HTMLElement; current = current.parentElement) ancestors.push(current);
      return ancestors;
    });
    const candidates = [...new Set([...dialogs, ...nearby])].filter(isUsableSurface);
    return candidates
      .map((element) => {
        const rect = element.getBoundingClientRect();
        const isDialog = element.matches('dialog, [role="dialog"], [aria-modal="true"]');
        const containsAction = actionX >= rect.left && actionX <= rect.right && actionY >= rect.top && actionY <= rect.bottom;
        const area = rect.width * rect.height;
        return { rect, score: (isDialog ? 1000000 : 0) + (containsAction ? 100000 : 0) - area };
      })
      .sort((a, b) => b.score - a.score)[0]?.rect;
  };

  const parkNearSurface = (actionX, actionY) => {
    const surface = findNearbySurface(actionX, actionY);
    if (!surface) {
      parkOnPageSurface();
      return;
    }
    const leftSpace = surface.left;
    const rightSpace = innerWidth - surface.right;
    const parkOnLeft = leftSpace >= rightSpace;
    const position = clampPosition({
      x: parkOnLeft ? surface.left - 18 : surface.right + 18,
      y: Math.max(surface.top + 64, Math.min(surface.bottom - 12, actionY))
    });
    startTravel(position, { mode: "run", duration_ms: 1150, returning: true });
  };

  const ready = app
    .init({
      canvas,
      width: CANVAS_WIDTH,
      height: CANVAS_HEIGHT,
      backgroundAlpha: 0,
      antialias: false,
      preference: "webgl"
    })
    .then(async () => {
      await Promise.all(
        Object.entries(assets).map(async ([name, config]) => {
          textures[name] = await makeFrames(config);
        })
      );
      sprite = new PIXI.AnimatedSprite(textures.idle);
      sprite.anchor.set(0.5, 1);
      sprite.scale.set(SCALE);
      sprite.position.set(SPRITE_X, SPRITE_FLOOR_Y);
      app.stage.addChild(sprite);
      setAnimation("idle", { restart: true });
      parkOnPageSurface({ animate: false });

      app.ticker.add(() => {
        const now = performance.now();
        const distance = Math.hypot(state.targetX - state.x, state.targetY - state.y);
        const moving = distance > 2 || Boolean(state.travel);
        if (moving) faceDirection(state.targetX - state.x);
        if (state.travel) {
          const progress = Math.min(1, (now - state.travel.startedAt) / state.travel.duration);
          const eased = easeInOut(progress);
          state.x = state.travel.startX + (state.travel.endX - state.travel.startX) * eased;
          state.y = state.travel.startY + (state.travel.endY - state.travel.startY) * eased;
          if (progress >= 1) {
            state.x = state.travel.endX;
            state.y = state.travel.endY;
            if (!nextTravelSegment()) state.travel = null;
          }
        } else {
          state.x += (state.targetX - state.x) * 0.06;
          state.y += (state.targetY - state.y) * 0.06;
        }
        placeCanvas();
        if (!moving && !state.returningToLedge) faceLookTarget();
        if (!moving && state.returningToLedge) {
          state.returningToLedge = false;
          faceDirection(state.x < innerWidth / 2 ? 1 : -1);
          canvas.style.pointerEvents = "auto";
          state.mode = "idle";
        }

        if (state.temporaryUntil && now > state.temporaryUntil) {
          state.temporaryUntil = 0;
          state.mode = moving ? "walk" : "idle";
        }
        if (!state.temporaryUntil) {
          const movementModes = new Set(["walk", "run", "jump", "runningJump"]);
          setAnimation(moving ? (movementModes.has(state.mode) ? state.mode : "walk") : state.mode);
        }
      });
    });

  const moveTo = async ({ x, y, label = "click", duration_ms = 600 }) => {
    await ready;
    window.clearTimeout(state.returnTimer);
    canvas.style.pointerEvents = "none";
    state.returningToLedge = false;
    state.lookAtX = x;
    state.lookAtY = y;
    const approachDirection = x >= state.x ? 1 : -1;
    const position = clampPosition({
      x: x - approachDirection * TARGET_SIDE_OFFSET,
      y: y + TARGET_VERTICAL_OFFSET
    });
    const travelBudget = Math.max(duration_ms - ARRIVAL_ACTION_LEAD_MS, MIN_TRAVEL_MS);
    const travelMs = startTravelPath(planTravelPath(position, { label, duration_ms: travelBudget }));
    window.setTimeout(() => {
      faceLookTarget();
      setTemporaryAnimation(label === "input" ? "idle" : "attack", 620);
    }, travelMs);
    state.returnTimer = window.setTimeout(() => {
      state.lookAtX = null;
      state.lookAtY = null;
      parkNearSurface(x, y);
    }, travelMs + TARGET_LINGER_MS);
  };

  document.documentElement.addEventListener("browser-use-pet-command", () => {
    const command = document.documentElement.getAttribute("data-browser-use-pet-command");
    if (!command) return;
    try {
      moveTo(JSON.parse(command));
    } finally {
      document.documentElement.removeAttribute("data-browser-use-pet-command");
    }
  });

  const jump = async () => {
    await ready;
    setTemporaryAnimation("jump", 620);
  };

  const success = async () => {
    await jump();
    window.setTimeout(() => {
      state.mode = "idle";
      setAnimation("idle");
    }, 700);
  };

  window.addEventListener("keydown", (event) => {
    const target = event.target;
    if (target instanceof HTMLElement && (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName))) return;
    if (event.code === "Digit1") {
      state.mode = "idle";
      setAnimation("idle");
    }
    if (event.code === "Digit2") {
      state.mode = "walk";
      state.targetX = clampPosition({ x: state.x - 170, y: state.y }).x;
      setAnimation("walk");
    }
    if (event.code === "Digit3") jump();
    if (event.code === "Digit4") setAnimation("idle");
    if (event.code === "Digit5") setAnimation("idle");
    if (event.code === "Digit6") setTemporaryAnimation("attack", 720);
    if (event.code === "Digit7") success();
  });

  window.addEventListener("resize", () => {
    const position = clampPosition(state);
    state.x = position.x;
    state.y = position.y;
    const target = clampPosition({ x: state.targetX, y: state.targetY });
    state.targetX = target.x;
    state.targetY = target.y;
    placeCanvas();
  });

  globalThis.__browserUsePetCompanion = {
    moveTo,
    jump,
    success,
    setMode(mode) {
      if (state.temporaryUntil > performance.now()) return;
      state.mode = mode === "thinking" || mode === "waiting" ? "idle" : mode;
    },
    goHome() {
      parkOnPageSurface();
    }
  };
})();
