# WeChat Mini Program

This directory contains a native WeChat Mini Program scaffold for `skin_alporithm`.

## Local development

1. Open `skin_alporithm/miniapp/` in WeChat DevTools.
2. Replace the placeholder AppID in `project.config.json` if you want real device preview.
3. Confirm the backend base URL in `miniapp/utils/config.js`.
   Current default: `http://192.168.1.23:5000`
4. In DevTools, disable strict domain verification for local LAN debugging.
   `project.private.config.json` already sets `urlCheck: false` for local use.
5. Start the Flask backend on the same machine and make sure the phone and computer are on the same LAN.

## Workflow

The mini program drives the backend in this order:

1. `POST /v1/mobile/upload-image`
2. `POST /v1/face-align`
3. `POST /v1/analyze-face`
4. Render summary cards, gallery images, and raw JSON

## Notes

- The mini program only talks to the Flask server.
- Result images are rendered through `/v1/mobile/result-image/<folder>/<filename>`.
- `utils/config.js` is the fastest place to change the LAN IP during development.
- If phone preview still blocks plain HTTP, use DevTools preview, configure request domains, or expose the backend through an HTTPS tunnel.
