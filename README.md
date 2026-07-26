# llm-browser-testkit

Describe browser tests in plain English. The LLM figures out which elements to
click and whether the page looks right.

```
llm-browser-testkit run smoke.toml
```

## Quick start

```bash
# Install
cargo install llm-browser-testkit

# Set your LLM credentials (OpenAI-compatible API)
export HARNESS_LLM_TEST_URL=https://api.openai.com
export HARNESS_LLM_TEST_MODEL=gpt-4o-mini
export HARNESS_LLM_API_KEY=sk-...

# Run the built-in example (tests example.com — no account needed)
llm-browser-testkit run examples/smoke.toml
```

## Write your first test

```toml
# hello.toml
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
```

Run it:

```bash
llm-browser-testkit run hello.toml
```

## Step reference

Every step has a `kind`. Required fields depend on the kind.

| `kind` | What it does | Required | Optional |
|--------|-------------|----------|----------|
| `navigate` | Open a URL | `url` | `wait_after_ms` |
| `click` | Click an element | `target` | `selector`, `wait_after_ms` |
| `type` | Type into a field | `target`, `text` | `selector`, `wait_after_ms` |
| `wait` | Wait for an element | `target` | `selector`, `timeout_ms` |
| `assert` | Check the page | one of `definition`, `preset`, or `prompt` | `assert_text` |
| `screenshot` | Save a .png | — | `path` |

**`target`** is natural language ("the submit button", "the search input"). The
LLM looks at the page DOM and picks the right CSS selector at runtime. Skip the
LLM with an explicit `selector`.

## Assertion presets

Built-in presets you can use inline or from `[[definitions]]`.

| Preset | What it checks |
|--------|---------------|
| `no_error_on_page` | No errors, stack traces, or broken UI on the page |
| `text_visible` | Specific text appears on the page (`assert_text`) |
| `element_exists` | A described UI element is present |

Custom assertions with `prompt` send any question to the LLM:

```toml
[[test.steps]]
kind = "assert"
prompt = "Does the page have a heading that says 'Example Domain'?"
```

## CLI reference

```
llm-browser-testkit run <scenario.toml> [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--llm-url` | `$HARNESS_LLM_TEST_URL` or `http://localhost:8080` | OpenAI-compatible endpoint |
| `--llm-model` | `$HARNESS_LLM_TEST_MODEL` or `deepseek` | Model name |
| `--llm-api-key` | `$HARNESS_LLM_API_KEY` | API key (Bearer token) |
| `--llm-header` | — | Custom header `Name:Value` (repeatable) |
| `--base-url` | `$HARNESS_BROWSER_BASE_URL` or `http://localhost:4200` | App under test |
| `--headless` | `true` | Run Chrome headlessly |
| `--timeout` | `60` | Seconds per action |
| `--viewport-width` | `1280` | Browser width |
| `--viewport-height` | `720` | Browser height |
| `--start-url` | `/dashboard` | First page to load |

CLI flags override the scenario `[config]`.

## How it works

Three pieces:

1. **Chrome** — launched via the [Chrome DevTools Protocol](https://chromedevtools.github.io/devtools-protocol/)
   (`headless_chrome` crate). It navigates, clicks, types, and extracts page
   content.

2. **LLM** — any OpenAI-compatible API. Used in two places:
   - **Element targeting**: when a step says `target = "the login button"`, the
     runner sends the page's interactive elements to the LLM and asks for a CSS
     selector.
   - **Assertions**: the runner sends page content to the LLM with a QA prompt
     and expects `PASS` or `FAIL: <reason>`.

3. **TOML scenarios** — declarative test files. No code, no CSS selectors
   required. Just describe what you want in English.

```
TOML file  →  CLI runner  →  Chrome (CDP)  →  LLM API
```

## Use as a library

```toml
[dependencies]
llm-browser-testkit = "0.1"
```

```rust
use llm_browser_testkit::runner::ScenarioRunner;
use llm_browser_testkit::scenario::Scenario;

let scenario: Scenario = toml::from_str(&contents)?;
let runner = ScenarioRunner::new(scenario.config.clone(), scenario.definitions);
let report = runner.run(&scenario.test)?;

println!("Passed: {}, Failed: {}", report.tests_passed, report.tests_failed);
```

### Macros: `#[browser_test]` in `cargo test`

Enable the `macros` feature to write browser tests directly in your Rust test
modules:

```toml
[dev-dependencies]
llm-browser-testkit = { version = "0.1", features = ["macros"] }
```

```rust,ignore
use llm_browser_testkit::browser_test;
use llm_browser_testkit::browser_test_inline;

// Run a TOML scenario file
browser_test!(homepage => "tests/homepage.toml");

// Inline small scenarios
browser_test_inline!(hello, r#"
[config]
base_url = "https://example.com"

[[test]]
name = "hello"

[[test.steps]]
kind = "navigate"
url = "/"

[[test.steps]]
kind = "assert"
preset = "no_error_on_page"
"#);
```

Tests auto-skip when no LLM endpoint or Chrome is available — safe to include in
every CI run. They only execute with real `PASS`/`FAIL` when infrastructure is
present.

### LLM authentication

The runner supports API keys and custom headers for SSO or alternative auth:

```toml
[config]
llm_api_key = "sk-..."
llm_headers = { "X-Org-ID" = "acme", "X-Project" = "qa" }
```

Via CLI:

```bash
llm-browser-testkit run tests.toml \
  --llm-api-key sk-... \
  --llm-header "X-Org-ID:acme" \
  --llm-header "X-Project:qa"
```

Via env:

```bash
export HARNESS_LLM_API_KEY=sk-...
export HARNESS_LLM_HEADERS='{"X-Org-ID":"acme","X-Project":"qa"}'
```

## License

Apache-2.0 OR MIT
