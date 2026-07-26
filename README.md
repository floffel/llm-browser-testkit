# llm-browser-testkit

LLM-driven browser test framework. Define browser test scenarios in TOML with
natural language steps — the LLM resolves element descriptions to CSS selectors
at runtime and evaluates assertions against page content.

## How it works

```
TOML scenario  →  CLI runner  →  Chrome (CDP)  →  LLM (OpenAI-compatible API)
```

1. **Write a TOML scenario** describing browser interactions in natural language
   (click "the Login button", type into "the search field", etc.)
2. **The CLI launches headless Chrome** via the Chrome DevTools Protocol
3. **For each interactive step**, the LLM receives the page's DOM (interactive
   elements) and resolves the natural language description to a CSS selector
4. **For each assertion**, the LLM evaluates the page content against the
   assertion prompt and responds PASS or FAIL

## Quick start

```bash
cargo install llm-browser-testkit

# Point it at your LLM endpoint (OpenAI-compatible API)
export HARNESS_LLM_TEST_URL=https://api.openai.com
export HARNESS_LLM_TEST_MODEL=gpt-4o-mini

# Run a scenario
llm-browser-testkit run examples/smoke.toml
```

## Scenario format

```toml
[config]
base_url = "https://example.com"
timeout_secs = 30
start_url = "/"

[[definitions]]
name = "no_errors"
preset = "no_error_on_page"

[[test]]
name = "Homepage loads"

[[test.steps]]
kind = "navigate"
url = "/"

[[test.steps]]
kind = "assert"
definition = "no_errors"

[[test.steps]]
kind = "click"
target = "the More information link"
wait_after_ms = 1000

[[test.steps]]
kind = "screenshot"
path = "result.png"
```

## Step kinds

| Kind | Description |
|------|-------------|
| `navigate` | Navigate to a URL (absolute or relative to `base_url`) |
| `click` | Click an element described in natural language |
| `type` | Type text into an input element |
| `wait` | Wait for an element to appear |
| `assert` | Evaluate an assertion (named definition, preset, or custom prompt) |
| `screenshot` | Capture a PNG screenshot |

## Built-in assertion presets

| Preset | Description |
|--------|-------------|
| `no_error_on_page` | Fails if the page contains errors, stack traces, or malfunctions |
| `text_visible` | Passes if the given text is found on the page |
| `element_exists` | Passes if the described UI element exists |

## Environment variables

| Variable | Default |
|----------|---------|
| `HARNESS_BROWSER_BASE_URL` | `http://localhost:4200` |
| `HARNESS_LLM_TEST_URL` | `http://localhost:8080` |
| `HARNESS_LLM_TEST_MODEL` | `deepseek` |
| `HARNESS_BROWSER_HEADLESS` | `true` |

## License

Apache-2.0 OR MIT
