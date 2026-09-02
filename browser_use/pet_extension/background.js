const bridgeUrl = "http://127.0.0.1:8765";
const deployedSitesKey = "browserUsePetDeployedSites";

const normalizeHostname = (hostname) => hostname.replace(/^www\./, "").toLowerCase();

const requestBridge = async (path, options = {}) => {
	const response = await fetch(`${bridgeUrl}${path}`, options);
	if (!response.ok) throw new Error(`Bridge returned ${response.status}`);
	return response.json();
};

const sessionIdForSender = (sender) => {
	if (sender?.tab?.id != null) return `tab:${sender.tab.id}`;
	throw new Error("Could not identify the Website Pet tab session.");
};

const getDeployTab = async (tab) => {
  if (tab?.id && tab.url) return tab;
  const [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return activeTab;
};

const deployCurrentSite = async (clickedTab) => {
  const tab = await getDeployTab(clickedTab);
  if (!tab?.id || !tab.url) return;
  const url = new URL(tab.url);
  if (url.protocol !== "http:" && url.protocol !== "https:") return;
  const hostname = normalizeHostname(url.hostname);
  const stored = await chrome.storage.local.get({ [deployedSitesKey]: [] });
  const deployedSites = Array.isArray(stored[deployedSitesKey])
    ? stored[deployedSitesKey].map(normalizeHostname)
    : [];
  if (!deployedSites.includes(hostname)) {
    await chrome.storage.local.set({ [deployedSitesKey]: [...deployedSites, hostname] });
  }
  await chrome.action.setBadgeText({ tabId: tab.id, text: "ON" });
  await chrome.action.setBadgeBackgroundColor({ tabId: tab.id, color: "#e9894c" });
  await chrome.tabs.reload(tab.id);
};

chrome.action.onClicked.addListener((tab) => {
  deployCurrentSite(tab).catch((error) => console.error("Website Pet deploy failed", error));
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
	const handle = async () => {
		const sessionId = sessionIdForSender(sender);
		switch (message.type) {
			case "pet:get-session":
				return { session_id: sessionId };
			case "pet:get-status":
				return requestBridge(`/status?session_id=${encodeURIComponent(sessionId)}`);
			case "pet:start-task":
				return requestBridge("/tasks", {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({
						session_id: sessionId,
						task: message.task,
						url: message.url,
						origin_token: message.origin_token
					})
				});
			case "pet:reply":
				return requestBridge("/reply", {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({ session_id: sessionId, reply: message.reply })
				});
			case "pet:stop":
				return requestBridge(`/stop?session_id=${encodeURIComponent(sessionId)}`, {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: "{}"
				});
      default:
        throw new Error(`Unknown Website Pet message: ${message.type}`);
    }
  };

  handle()
    .then((data) => sendResponse({ ok: true, data }))
    .catch((error) => sendResponse({ ok: false, error: error.message }));
  return true;
});
