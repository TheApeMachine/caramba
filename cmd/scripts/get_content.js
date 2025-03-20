() => {
  try {
    // Start with the document body if we're in a browser context
    const source = document?.body || null;

    // Early exit if no source available
    if (!source) {
      return "No content source available to extract from.";
    }

    // Work with a clone to avoid modifying the original
    let tempDoc;
    try {
      tempDoc = source.cloneNode(true);
    } catch (e) {
      // If cloning fails, work with the original
      tempDoc = source;
    }

    // Remove elements that typically don't contain interesting content
    const elementsToRemove = [
      "script",
      "style",
      "noscript",
      "iframe",
      "svg",
      "path",
      "header",
      "footer",
      "nav",
      "aside",
      "banner",
      "ad",
      "advert",
      "cookie",
      "popup",
      "modal",
      "dialog",
      "embed",
      "object",
      "form",
      "input",
      "button",
      "comment",
      "advertisement",
    ];

    try {
      elementsToRemove.forEach((tag) => {
        try {
          const elements = tempDoc.querySelectorAll(tag);
          if (elements && elements.length) {
            elements.forEach((el) => {
              try {
                el.parentNode?.removeChild(el);
              } catch (e) {
                // Silent fail for individual element removal
              }
            });
          }
        } catch (e) {
          // Skip this tag if it causes issues
        }
      });
    } catch (e) {
      // Continue even if element removal fails
    }

    // Remove elements with certain classes/ids
    const selectorsToRemove = [
      '[class*="nav"]',
      '[class*="menu"]',
      '[class*="header"]',
      '[class*="footer"]',
      '[class*="sidebar"]',
      '[class*="widget"]',
      '[class*="banner"]',
      '[class*="ad-"]',
      '[class*="cookie"]',
      '[class*="popup"]',
      '[id*="nav"]',
      '[id*="menu"]',
      '[id*="header"]',
      '[id*="footer"]',
      '[id*="sidebar"]',
      '[id*="ad-"]',
      '[id*="cookie"]',
      '[id*="popup"]',
      '[aria-hidden="true"]',
      '[role="banner"]',
      '[role="navigation"]',
      '[role="complementary"]',
      '[role="contentinfo"]',
    ];

    try {
      selectorsToRemove.forEach((selector) => {
        try {
          const elements = tempDoc.querySelectorAll(selector);
          if (elements && elements.length) {
            elements.forEach((el) => {
              try {
                el.parentNode?.removeChild(el);
              } catch (e) {
                // Silent fail for individual element removal
              }
            });
          }
        } catch (e) {
          // Skip this selector if it causes issues
        }
      });
    } catch (e) {
      // Continue even if selector removal fails
    }

    // Find potential main content areas
    let mainContent = null;
    const mainSelectors = [
      "main",
      '[role="main"]',
      "article",
      ".content",
      "#content",
      ".article",
      "#article",
      ".post",
      "#post",
      ".page",
      "#page",
    ];

    for (const selector of mainSelectors) {
      try {
        const element = tempDoc.querySelector(selector);
        if (element && element.textContent.trim().length > 100) {
          mainContent = element;
          break;
        }
      } catch (e) {
        // Continue trying other selectors
      }
    }

    // If still no main content, use the body or tempDoc itself
    if (!mainContent) {
      mainContent = tempDoc.body || tempDoc;
    }

    // If we still don't have content, return a message
    if (!mainContent) {
      return "Could not identify content in this page.";
    }

    // Gather all text content elements
    let contentElements = [];
    const elementTypes = [
      "p",
      "h1",
      "h2",
      "h3",
      "h4",
      "h5",
      "h6",
      "li",
      "blockquote",
      "pre",
      "code",
      "figcaption",
    ];

    try {
      elementTypes.forEach((type) => {
        try {
          const elements = mainContent.querySelectorAll(type);
          if (elements && elements.length) {
            contentElements = [...contentElements, ...Array.from(elements)];
          }
        } catch (e) {
          // Skip this element type if it causes issues
        }
      });
    } catch (e) {
      // If structured extraction fails, try to get all text directly
      return extractTextFallback(mainContent);
    }

    // If no elements found through selectors, use fallback
    if (contentElements.length === 0) {
      return extractTextFallback(mainContent);
    }

    // Filter out elements with very little text or that appear hidden
    const significantElements = contentElements.filter((el) => {
      try {
        // Check if element has reasonable text length
        const text = el.textContent?.trim() || "";
        if (text.length < 10 && !el.tagName.match(/^H[1-6]$/i)) {
          return false;
        }

        // Basic visibility check - note getComputedStyle might not be available in all contexts
        try {
          if (typeof window !== "undefined" && window.getComputedStyle) {
            const style = window.getComputedStyle(el);
            if (
              style.display === "none" ||
              style.visibility === "hidden" ||
              style.opacity === "0"
            ) {
              return false;
            }
          }
        } catch (e) {
          // Skip style check if not available
        }

        return true;
      } catch (e) {
        return false;
      }
    });

    // Build the output with proper formatting
    let result = "";
    let previousWasHeading = false;

    significantElements.forEach((el) => {
      try {
        // Get the text of the current element
        let text = el.textContent?.trim() || "";

        // Skip empty elements
        if (!text) return;

        // Replace multiple whitespaces with a single space
        text = text.replace(/\s+/g, " ");

        // Format based on element type
        const tagName = el.tagName.toUpperCase();

        if (tagName.match(/^H[1-6]$/)) {
          // Add extra newline before headings (unless it's the first element)
          if (result.length > 0 && !previousWasHeading) {
            result += "\n\n";
          }
          result += text + "\n";
          previousWasHeading = true;
        } else if (tagName === "LI") {
          result += "• " + text + "\n";
          previousWasHeading = false;
        } else if (tagName === "BLOCKQUOTE") {
          result += "> " + text + "\n\n";
          previousWasHeading = false;
        } else if (tagName === "PRE" || tagName === "CODE") {
          result += "```\n" + text + "\n```\n\n";
          previousWasHeading = false;
        } else {
          if (previousWasHeading) {
            result += text + "\n\n";
          } else {
            // Add double newline between paragraphs
            result += (result.length > 0 ? "\n\n" : "") + text;
          }
          previousWasHeading = false;
        }
      } catch (e) {
        // Skip this element if processing fails
      }
    });

    // Clean up extra whitespace
    result = result.replace(/\n{3,}/g, "\n\n");

    // Fallback function for direct text extraction if needed
    function extractTextFallback(element) {
      try {
        let text = element.textContent || "";
        text = text.replace(/\s+/g, " ").trim();

        // Split into paragraphs on double line breaks or significant punctuation followed by space
        const paragraphs = text
          .split(/\n\n+|(?<=[.!?])\s+(?=[A-Z])/g)
          .filter((p) => p.trim().length > 20) // Only paragraphs with meaningful content
          .map((p) => p.trim());

        return paragraphs.join("\n\n");
      } catch (e) {
        return "Failed to extract text content.";
      }
    }

    // Return empty string if no meaningful content was found
    if (!result.trim()) {
      return "No meaningful content found on this page.";
    }

    return result.trim();
  } catch (e) {
    // Return error message if something goes wrong
    return `Error extracting content: ${e.message}`;
  }
};
