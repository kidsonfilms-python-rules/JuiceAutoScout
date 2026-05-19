# Contributing

Thanks for contributing to JuiceAutoScout.

This repository mixes Python tracking code, browser-based tooling, and documentation. The most helpful contributions are focused, well-explained, and validated against the part of the workflow they change.

## Repository Areas

- `auto_scout.py`
  Main tracker CLI, exports, logging, and shot detection.
- `util/juice_log.py`
  Shared JUICE LOG schema implementation for compact pose/shot logs.
- `tools/calibrate.py`
  Manual field-corner calibration tool.
- `tools/manual_tracker.html`
  Browser-based manual labeling tool.
- `util/jlog.js`
  Shared browser-side JLOG encoder/decoder used by the HTML tools.
- `tools/debug.py`
  WPILOG inspection helper.
- `tools/data_visualizer.html`, `tools/shot_visualizer.html`
  Browser-based analysis and visualization helpers.
- `README.md`, `DOCUMENTATION.md`
  User-facing docs. Update these when behavior, flags, outputs, or file locations change.

## Getting Set Up

Install the main runtime dependencies:

```bash
pip install opencv-python numpy progress scipy
```

Some workflows also need:

- `yt-dlp` for YouTube downloads
- a local video file for tracker or calibration testing

## Development Expectations

- Keep pull requests focused. Avoid bundling unrelated fixes.
- Prefer small, targeted changes over broad rewrites.
- Preserve existing file formats unless the change intentionally updates them.
- If you change user-visible behavior, update the docs in the same PR.
- If you move files or tools, update paths in both `README.md` and `DOCUMENTATION.md`.

## Validation

Run the validation that matches your change.

For Python changes, at minimum:

```bash
python3 -m py_compile auto_scout.py util/juice_log.py tools/calibrate.py tools/debug.py
```

If your change affects tracking, calibration, or exports, also do a realistic manual check when possible:

- run `auto_scout.py` on a local example video
- verify CSV, JLOG, and WPILOG outputs are created as expected
- check any affected debug output
- open the browser tools if you changed their UI or data flow

In your PR description, mention what you validated and what you did not validate.

## Data and Generated Files

Be careful with local and generated artifacts.

- Do not commit `output/` contents unless the change explicitly requires a checked-in example artifact.
- Do not commit personal or temporary calibration files unless they are intentional shared examples.
- Do not commit large downloaded videos unintentionally.
- Keep example data additions small and clearly justified.

## Style Notes

- Follow the existing coding style in the file you are editing.
- Keep comments brief and only where they help explain non-obvious logic.
- Prefer explicit, readable logic over compact cleverness in tracker code.
- For browser tools, keep them no-build and easy to open locally.

## Pull Requests

A good PR includes:

- a short summary of what changed
- why the change was needed
- any relevant screenshots, terminal output, or sample results for UI/output changes
- a brief validation note

If there are tradeoffs or known limitations, call them out directly.

## Documentation Changes

Please update docs when you change:

- CLI flags or defaults
- output files or formats
- coordinate conventions
- tool locations or repository structure
- workflow steps for calibration, manual labeling, or visualization

## Questions

If you are unsure whether a change belongs in this repo, open a small issue or PR with the proposed scope and rationale before doing a larger refactor.
