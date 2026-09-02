(() => {
  const storageKey = "browserUsePetDeployedSites";
  const normalizeHostname = (hostname) => hostname.replace(/^www\./, "").toLowerCase();
  const currentHost = normalizeHostname(location.hostname);
  const matchesSite = (hostname, site) => hostname === site || hostname.endsWith(`.${site}`);

  const ready = (async () => {
    try {
      const stored = await chrome.storage.local.get({ [storageKey]: [] });
      const deployedSites = Array.isArray(stored[storageKey]) ? stored[storageKey].map(normalizeHostname) : [];
      const enabled = deployedSites.some((site) => matchesSite(currentHost, site));
      document.documentElement.toggleAttribute("data-browser-use-pet-site-native-enabled", enabled);
      return { enabled, currentHost, deployedSites };
    } catch {
      document.documentElement.removeAttribute("data-browser-use-pet-site-native-enabled");
      return { enabled: false, currentHost, deployedSites: [] };
    }
  })();

  globalThis.__browserUsePetSiteNative = {
    storageKey,
    currentHost,
    matchesSite,
    normalizeHostname,
    ready
  };
})();
