// Runs at document_start — BEFORE the page's own scripts — so our key listener is registered first
// and fires before the page's (e.g. a game's arrow-key handler). When a keystroke originates from
// inside the pet UI, we stop it from reaching the page so typing in the pet box can't drive the page
// (e.g. move 2048 tiles). We never preventDefault, so the character still types into the pet's field.
// The pet UI (id "browser-use-pet-agent") is created later by pet.js; we look it up at event time.
(() => {
  const PET_HOST_ID = "browser-use-pet-agent";

  const isFromPet = (event) => {
    const host = document.getElementById(PET_HOST_ID);
    if (!host) return false;
    const path = typeof event.composedPath === "function" ? event.composedPath() : [];
    // path.includes(host): the key was dispatched from inside the pet's shadow tree.
    // host.contains(activeElement): focus is inside the pet (activeElement is the shadow host).
    return path.includes(host) || host.contains(document.activeElement);
  };

  const trap = (event) => {
    if (isFromPet(event)) event.stopImmediatePropagation();
  };

  for (const type of ["keydown", "keypress", "keyup"]) {
    window.addEventListener(type, trap, true);
  }
})();
