"""Styles for the Caramba TUI.

This module contains all the CSS styling for the terminal user interface,
providing a cohesive and modern design language.
"""
from __future__ import annotations

# Color palette reference (for documentation):
# primary: #7C3AED
# primary-light: #A78BFA
# primary-dark: #5B21B6
# secondary: #10B981
# accent: #F59E0B
# error: #EF4444
# warning: #F59E0B
# success: #10B981
# info: #3B82F6
# surface: #1E1E2E
# surface-light: #2D2D3D
# surface-lighter: #3D3D4D
# background: #11111B
# text: #CDD6F4
# text-muted: #6C7086
# border: #45475A

# Main application styles
TUI_CSS = """
/* ─────────────────────────────────────────────────────────────────
   Base Screen Layout
   ───────────────────────────────────────────────────────────────── */
Screen {
    background: #11111B;
    layout: horizontal;
}

/* ─────────────────────────────────────────────────────────────────
   Header & Footer
   ───────────────────────────────────────────────────────────────── */
Header {
    background: #7C3AED;
    color: #CDD6F4;
    dock: top;
    text-style: bold;
}

Footer {
    background: #1E1E2E;
    color: #6C7086;
    dock: bottom;
}

FooterKey > .footer-key--key {
    background: #7C3AED;
    color: #CDD6F4;
}

FooterKey > .footer-key--description {
    color: #6C7086;
}

/* ─────────────────────────────────────────────────────────────────
   Main Layout Containers
   ───────────────────────────────────────────────────────────────── */
#main-container {
    width: 100%;
    height: 100%;
    layout: horizontal;
}

#left-sidebar {
    width: 32;
    background: #1E1E2E;
    border-right: solid #45475A;
    padding: 0;
}

#right-sidebar {
    width: 32;
    background: #1E1E2E;
    border-left: solid #45475A;
    padding: 0;
}

#chat-area {
    width: 1fr;
    height: 100%;
    layout: vertical;
    background: #11111B;
}

/* ─────────────────────────────────────────────────────────────────
   Sidebar Styling
   ───────────────────────────────────────────────────────────────── */
.sidebar-header {
    height: 3;
    background: #2D2D3D;
    padding: 1 2;
    text-style: bold;
    color: #CDD6F4;
    border-bottom: solid #45475A;
}

.sidebar-content {
    height: 1fr;
    overflow-y: auto;
    padding: 1;
}

.sidebar-item {
    height: auto;
    padding: 0 1;
    margin-bottom: 1;
}

.sidebar-item:hover {
    background: #2D2D3D;
}

/* Expert Status Indicators */
.expert-status {
    padding: 0 1;
    margin-bottom: 1;
}

.expert-idle {
    color: #6C7086;
}

.expert-consulting {
    color: #F59E0B;
    text-style: bold;
}

.expert-done {
    color: #10B981;
}

.expert-error {
    color: #EF4444;
}

/* Tool Call Items */
.tool-call {
    padding: 1;
    margin-bottom: 1;
    background: #2D2D3D;
    border: round #45475A;
}

.tool-call-name {
    color: #3B82F6;
    text-style: bold;
}

.tool-call-agent {
    color: #6C7086;
}

/* ─────────────────────────────────────────────────────────────────
   Chat Viewport
   ───────────────────────────────────────────────────────────────── */
#chat-viewport {
    height: 1fr;
    background: #11111B;
    padding: 1 2;
    overflow-y: auto;
}

/* Message Styling */
.message {
    width: 100%;
    padding: 1 2;
    margin-bottom: 1;
}

.message-user {
    background: #7C3AED;
    color: #CDD6F4;
    border: round #5B21B6;
    margin-left: 8;
}

.message-assistant {
    background: #1E1E2E;
    color: #CDD6F4;
    border: round #45475A;
    margin-right: 8;
}

.message-system {
    background: #2D2D3D;
    color: #6C7086;
    text-align: center;
    text-style: italic;
    border: none;
}

.message-error {
    background: #3D1F1F;
    color: #EF4444;
    border: round #EF4444;
}

.message-header {
    color: #6C7086;
    margin-bottom: 1;
}

.message-content {
    color: #CDD6F4;
}

.message-timestamp {
    color: #6C7086;
    text-align: right;
}

/* ─────────────────────────────────────────────────────────────────
   Input Area
   ───────────────────────────────────────────────────────────────── */
#input-container {
    height: auto;
    min-height: 5;
    max-height: 12;
    background: #1E1E2E;
    padding: 1 2;
    border-top: solid #45475A;
}

#input-wrapper {
    height: auto;
    layout: horizontal;
    background: #2D2D3D;
    border: round #45475A;
    padding: 0 1;
}

#input-wrapper:focus-within {
    border: round #7C3AED;
}

#chat-input {
    background: transparent;
    border: none;
    height: auto;
    min-height: 1;
    padding: 1;
    color: #CDD6F4;
}

#chat-input:focus {
    border: none;
}

#chat-input .input--placeholder {
    color: #6C7086;
}

#send-button {
    width: auto;
    min-width: 8;
    height: 3;
    background: #7C3AED;
    color: #CDD6F4;
    border: none;
    margin-left: 1;
}

#send-button:hover {
    background: #A78BFA;
}

#send-button:focus {
    background: #5B21B6;
}

/* ─────────────────────────────────────────────────────────────────
   Autocomplete Dropdown
   ───────────────────────────────────────────────────────────────── */
#autocomplete-container {
    height: auto;
    max-height: 16;
    display: none;
    background: #1E1E2E;
    border: round #45475A;
    margin-bottom: 1;
    padding: 0;
}

#autocomplete-container.visible {
    display: block;
}

.autocomplete-item {
    height: 3;
    padding: 0 2;
    background: transparent;
    color: #CDD6F4;
}

.autocomplete-item:hover {
    background: #2D2D3D;
}

.autocomplete-item.selected {
    background: #7C3AED;
    color: #CDD6F4;
}

.autocomplete-command {
    color: #A78BFA;
    text-style: bold;
}

.autocomplete-description {
    color: #6C7086;
    margin-left: 2;
}

/* ─────────────────────────────────────────────────────────────────
   Command Palette Modal
   ───────────────────────────────────────────────────────────────── */
#command-palette {
    width: 60%;
    max-width: 80;
    height: auto;
    max-height: 24;
    background: #1E1E2E;
    border: round #7C3AED;
    padding: 1;
    margin: 4 0;
}

#command-palette-input {
    width: 100%;
    background: #2D2D3D;
    border: round #45475A;
    padding: 1;
    margin-bottom: 1;
}

#command-palette-input:focus {
    border: round #7C3AED;
}

#command-palette-results {
    height: auto;
    max-height: 16;
    overflow-y: auto;
}

.command-item {
    height: 3;
    padding: 0 2;
    background: transparent;
}

.command-item:hover {
    background: #2D2D3D;
}

.command-item.selected {
    background: #7C3AED;
}

.command-name {
    color: #A78BFA;
    text-style: bold;
}

.command-shortcut {
    color: #6C7086;
    text-align: right;
}

/* ─────────────────────────────────────────────────────────────────
   Status Indicators
   ───────────────────────────────────────────────────────────────── */
.status-bar {
    height: 1;
    background: #1E1E2E;
    padding: 0 2;
    border-top: solid #45475A;
}

.status-item {
    color: #6C7086;
    margin-right: 2;
}

.status-connected {
    color: #10B981;
}

.status-disconnected {
    color: #EF4444;
}

.status-loading {
    color: #F59E0B;
}

/* ─────────────────────────────────────────────────────────────────
   Loading / Spinner
   ───────────────────────────────────────────────────────────────── */
.thinking-indicator {
    color: #A78BFA;
    text-style: italic;
    padding: 1;
}

.dots {
    color: #F59E0B;
}

/* ─────────────────────────────────────────────────────────────────
   Scrollbars
   ───────────────────────────────────────────────────────────────── */
VerticalScroll {
    scrollbar-background: #2D2D3D;
    scrollbar-color: #6C7086;
    scrollbar-color-hover: #A78BFA;
    scrollbar-color-active: #7C3AED;
}

/* ─────────────────────────────────────────────────────────────────
   Tooltips / Help
   ───────────────────────────────────────────────────────────────── */
.tooltip {
    background: #2D2D3D;
    color: #CDD6F4;
    padding: 1 2;
    border: round #45475A;
}

/* ─────────────────────────────────────────────────────────────────
   Markdown Content (in messages)
   ───────────────────────────────────────────────────────────────── */
Markdown {
    background: transparent;
}

Markdown H1 {
    color: #A78BFA;
    text-style: bold;
}

Markdown H2 {
    color: #A78BFA;
    text-style: bold;
}

MarkdownFence {
    background: #2D2D3D;
    border: round #45475A;
    padding: 1;
}

MarkdownFence > .code_inline {
    background: #2D2D3D;
    color: #10B981;
}
"""
