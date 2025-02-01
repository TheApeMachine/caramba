() => {
  const out = new Set();

  const consider = [
    "article",
    "main",
    "[role='main']",
    ".content",
    "#content",
    ".post",
    ".article",
  ];

  let mainContent = document.body;

  for (const selector of consider) {
    const container = document.querySelector(selector);
    if (container) {
      mainContent = container;
    }
  }

  for (const element of mainContent.querySelectorAll(
    "p, h1, h2, h3, h4, h5, h6"
  )) {
    out.add(
      element.innerText
        .split("\n")
        .map((line) => line.trim())
        .filter((line) => line.length > 5)
        .join("\n")
    );
  }

  return Array.from(out.values()).join("\n");
};
