() => {
  const out = new Set();
  const currentURL = window.location.href.split("?")[0];
  const ignore = [
    "#",
    "javascript:",
    "mailto:",
    "tel:",
    "whatsapp:",
    "sms:",
    "terms",
    "privacy",
    "cookie",
    "login",
    "register",
    "forgot",
    "reset",
  ];

  const consider = [
    "article",
    "main",
    "[role='main']",
    ".content",
    "#content",
    ".post",
    ".article",
  ];

  // Attempt to find the main content container.
  let mainContent = document.body;

  for (const selector of consider) {
    const container = document.querySelector(selector);
    if (container && container.children.length > 2) {
      mainContent = container;
      break;
    }
  }

  for (const link of mainContent.querySelectorAll("a")) {
    if (link.href.startsWith(currentURL)) {
      continue;
    }

    if (ignore.some((ignore) => link.href.includes(ignore))) {
      continue;
    }

    out.add(link.href);
  }

  return Array.from(out.values()).join("\n");
};
