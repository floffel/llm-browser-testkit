# llm-browser-testkit

Describe browser tests in plain English. The LLM figures out which elements to
click and whether the page looks right — plus A2A agents, MCP tool-calling, cost
tracking, and budgets.

```
llm-browser-testkit run smoke.toml
```

## Contents

- [Quick start](#quick-start)
- [Write your first test](#write-your-first-test)
- [Step reference](#step-reference)
- [Assertion presets](#assertion-presets)
- [CLI reference](#cli-reference)
- [Endpoints](#endpoints)
- [A2A agents](#a2a-agents)
- [Run as an A2A agent](#run-as-an-a2a-agent)
- [MCP tools](#mcp-tools)
- [MCP server](#mcp-server-exposure)
- [Cost tracking & budgets](#cost-tracking--budgets)
- [How it works](#how-it-works)
- [Use as a library](#use-as-a-library)
- [LLM authentication](#llm-authentication)
- [License](#license)

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
| `login` | Idempotent login: fills the form if present, passes silently when already authenticated | `email`, `password` | `url` (default `/auth/login`), `wait_after_ms` |
| `click` | Click an element | `target` | `selector`, `wait_after_ms`, `endpoint` |
| `type` | Type into a field | `target`, `text` | `selector`, `wait_after_ms`, `endpoint` |
| `wait` | Wait for an element and/or visible text | `target` | `selector`, `text`, `timeout_ms`, `endpoint` |
| `assert` | Check the page | one of `definition`, `preset`, or `prompt` | `assert_text`, `endpoint` |
| `screenshot` | Save a .png | — | `path` |
| `agent` | Call an A2A agent | `agent`, `task` | `definition` |
| `mcp` | Call an MCP tool | `server`, `tool` | `args` |

**`target`** is natural language ("the submit button", "the search input"). The
LLM looks at the page DOM and picks the right CSS selector at runtime. Skip the
LLM with an explicit `selector`.

**`login`** — navigate to the login page and authenticate. If the app is
already signed in (no login form rendered), the step passes silently, so
scenarios that run the same test across a viewport matrix — or repeat login
steps in one browser session — stay green:

```toml
[[test.steps]]
kind = "login"
url = "/auth/login"
email = "admin@example.com"
password = "correct horse battery staple"
```

The step types into `#email` / `#password`, waits for a bot-protection token
(`input[name="cf-turnstile-response"][value]:not([value=""])`, up to 30s),
clicks `button.btn--landing.btn--primary` and waits for the authenticated
shell (`app-account-shell`, up to 30s).

**`endpoint`** routes this step to a specific [endpoint](#endpoints). Use it to
send element targeting to one model and assertions to another.

**`wait` with `text`** waits until the page's visible text contains a
substring — no selector or LLM needed:

```toml
[[test.steps]]
kind = "wait"
target = "the success message"
text = "Welcome back"
timeout_ms = 5000
```

Set both `selector` and `text` to require both conditions. The combined wait
shares one `timeout_ms` budget.

## Failure diagnostics & artifacts

When a step fails, the runner captures the current page state and writes a
screenshot, so CI logs answer *why* the step failed instead of printing a bare
timeout:

```
❌ [wait] the authenticated shell — wait for app-account-shell timed out after
30000ms: The event waited for never came — page: http://127.0.0.1:8082/auth/login
(Immosai) — visible: "Email address ⏎ Password ⏎ Sign in"
    │ url:      http://127.0.0.1:8082/auth/login
    │ title:    Immosai — Anmeldung
    │ content:  Email address  Password  ...  Invalid credentials.
    📸 screenshot: artifacts/account__login-and-open-account__006-wait.png
```

- **Page state** — URL, title, visible text, and any alert/error elements
  (`[role="alert"]`, `.error-message`, snackbars, …) are appended to the step
  message and printed in full to stderr.
- **Screenshots** — one PNG per failed step, written under
  `--artifacts-dir` (default `artifacts/`, env `HARNESS_ARTIFACTS_DIR`).
- **Fail fast** — by default the first failed step ends the test and the
  remaining steps are reported as skipped (no LLM budget is burned asserting
  against a page that is already known broken). Set
  `continue_on_failure = true` in `[config]` or pass `--continue-on-failure`
  to keep executing every step.
- **LLM element targeting is verified** — a response that is not a selector
  (`:not(*)`, `null`, explanations, …) fails immediately with the raw LLM
  output; a selector that matches nothing triggers one retry with feedback.
- **LLM errors are specific** — HTTP status, a truncated response-body
  snippet, and the attempt count are included, and deterministic client
  errors (401/403/404) fail fast instead of burning three retries.
- **Assertions always see the page** — custom `system`+`user_template`
  definitions that omit the `{content}` placeholder automatically get the
  page URL/title/content appended, so the LLM never answers "I can't
  determine that without seeing the page".

## Vision assertions (screenshots)

Text/DOM evaluation cannot see *how* the page renders — overlapping elements,
clipped text, or a cookie banner covering the content are invisible to
`innerText`. Mark an endpoint as vision-capable and attach a screenshot to an
assert step to let the LLM evaluate the actual pixels:

```toml
[config]                      # optional: cap the screenshot resolution
screenshot_max_dimension = 1400

[config.endpoints.vision]     # MUST declare vision = true
type = "llm"
url = "https://api.openai.com"
model = "gpt-4o"
api_key = "sk-..."
vision = true                 # ← the flag
pricing = { input_per_1m_tokens = 2.50, output_per_1m_tokens = 10.00 }

[[definitions]]
name = "no_overlaps"
preset = "visual_no_overlaps"

[[test.steps]]
kind = "assert"
definition = "no_overlaps"
endpoint = "vision"
screenshot = true             # ← attach the viewport screenshot
```

How it works:

- The viewport is captured as PNG, downscaled in Rust (Lanczos) so its
  longest edge is at most `screenshot_max_dimension` (default 1400), and
  re-encoded as quality-85 JPEG — no page JS, deterministic, and cheap on
  vision tokens.
- The image is sent as an OpenAI-compatible `image_url` content part next to
  the text prompt (which still includes the page text for context).
- Built-in presets: `visual_no_issues`, `visual_no_overlaps`,
  `visual_text_visible` (uses `assert_text`). Custom `screenshot = true`
  prompts work too.
- A `screenshot = true` step that resolves to an endpoint without
  `vision = true` fails immediately with a clear configuration error.
- Text-only workflows are untouched: without `screenshot = true` the
  request keeps the plain string `content` shape.

See [`examples/visual-overlays.toml`](examples/visual-overlays.toml) + the
bundled [`examples/visual-test-page.html`](examples/visual-test-page.html)
fixture for a runnable demo that passes on a clean page and detects a cookie
banner overlay. The prompts of the visual presets are intentionally strict
("only fail on clearly visible, user-impacting defects") — tune them per app
if your overlay detection needs to be more or less sensitive.

## DOM layout assertions (`layout_no_issues`)

Vision models see pixels but cost money per page × viewport. For cheap,
deterministic layout coverage there is a DOM-only preset that never calls
the LLM:

```toml
[[test.steps]]
kind = "assert"
preset = "layout_no_issues"   # no endpoint, no screenshot, no tokens
```

It evaluates a geometry scan in the page and fails with the detected issues:

- **page-overflow-x** — the document is wider than the viewport
  (horizontal scrolling or a runaway element);
- **element-out-of-viewport** — a visible, non-fixed element sticks out of
  the right/bottom viewport edge while still partially on screen;
- **text-clipped** — content inside an `overflow: hidden` container is
  measurably larger than the box (cut-off text);
- **element-overlap** — an interactive element's center point is covered
  by a different element that would intercept the click.

Intentional stacking (off-canvas drawers, dropdowns, badges, fixed headers,
fully-offscreen scroll content) is excluded by position/relation filters.
Run it after every page load — it is free, so it is also the perfect
companion for the viewport matrix below.

## Viewport matrix (mobile / tablet / desktop)

`[config.viewport_matrix]` expands **every test** in a scenario into one
variant per named viewport. Each variant overrides the browser viewport via
CDP device-metrics emulation and gets a ` — <name>` suffix on the test name;
per-test budgets apply per variant.

```toml
[config.viewport_matrix]
viewports = [
  { name = "mobile",  width = 390,  height = 844 },
  { name = "tablet",  width = 768,  height = 1024 },
  { name = "desktop", width = 1280, height = 720 },
]

[[test]]
name = "Dashboard renders"
steps = [
    { kind = "navigate", url = "/dashboard", wait_after_ms = 2000 },
    { kind = "assert", preset = "layout_no_issues" },
]
```

The above runs "Dashboard renders — mobile", "— tablet" and "— desktop",
each at its viewport, and the layout scan flags sticky overlays, off-screen
text, or covered controls per size. Use it with `screenshot = true` +
`visual_no_issues` on a vision endpoint for pixel-level checks on top.

Single-test overrides work too — any `[[test]]` may set
`viewport_width` / `viewport_height` directly, which also switches the
browser viewport via CDP for just that test:

```toml
[[test]]
name = "Narrow phone layout"
viewport_width = 320
viewport_height = 568
```

## Assertion presets

Built-in presets you can use inline or from `[[definitions]]`.

| Preset | What it checks |
|--------|---------------|
| `no_error_on_page` | No errors, stack traces, or broken UI on the page |
| `text_visible` | Specific text appears on the page (`assert_text`) |
| `element_exists` | A described UI element is present |
| `layout_no_issues` | **DOM scan, no LLM**: page overflow, elements out of viewport, clipped text, covered controls |
| `visual_no_issues` | **Screenshot**: no layout/rendering defects (overlaps, clipping, cut-off content, broken images, blank panels) |
| `visual_no_overlaps` | **Screenshot**: no elements covering other content or intercepting clicks |
| `visual_text_visible` | **Screenshot**: `assert_text` is fully visible and readable (not clipped or covered) |

Custom assertions with `prompt` send any question to the LLM:

```toml
[[test.steps]]
kind = "assert"
prompt = "Does the page have a heading that says 'Example Domain'?"
```

Custom presets with `system` + `user_template` let you define reusable assertion
logic with template variables `{url}`, `{title}`, `{content}`, `{expected_text}`,
and `{description}`. Forgetting `{content}` is no longer a problem — the page
context is appended automatically whenever the template does not reference it:

```toml
[[definitions]]
name = "text_matches"
system = "You are a QA tester."
user_template = "Does the page at {url} contain the text: {expected_text}?"
assert_text = "Welcome back"
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
| `--model-param` | — | Provider param `key=value` (repeatable) |
| `--base-url` | `$HARNESS_BROWSER_BASE_URL` or `http://localhost:4200` | App under test |
| `--headless` | `true` | Run Chrome headlessly |
| `--timeout` | `60` | Seconds per action |
| `--viewport-width` | `1280` | Browser width |
| `--viewport-height` | `720` | Browser height |
| `--start-url` | `/dashboard` | First page to load |
| `--max-cost` | — | Global budget: max USD across all tests |
| `--max-tokens` | — | Global budget: max tokens across all tests |
| `--budget-enforcement` | `hard` | Budget mode: `hard` (abort) or `soft` (warn) |
| `--artifacts-dir` | `$HARNESS_ARTIFACTS_DIR` or `artifacts` | Directory for failure screenshots |
| `--continue-on-failure` | off | Keep running remaining steps after a step failure (default: fail fast) |

CLI flags override the scenario `[config]`.

## Endpoints

Define multiple named endpoints — LLM providers, MCP servers, and A2A agents —
each with their own pricing, and route test steps to them automatically or
explicitly.

```toml
[config.endpoints.default]
type = "llm"
url = "https://api.openai.com"
model = "gpt-4o-mini"
api_key = "sk-..."
pricing = { input_per_1m_tokens = 0.15, output_per_1m_tokens = 0.60 }
default_for = ["targeting", "assertion"]

[config.endpoints.vision]
type = "llm"
url = "https://api.openai.com"
model = "gpt-4o"
api_key = "sk-..."
pricing = { input_per_1m_tokens = 2.50, output_per_1m_tokens = 10.00 }
default_for = []

[config.endpoints.db_mcp]
type = "mcp"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"]
pricing = { per_call = 0.001 }

[config.endpoints.audit_agent]
type = "a2a"
url = "http://localhost:9090"
pricing = { per_call = 0.01 }

[[test]]
name = "Dashboard with vision"

[[test.steps]]
kind = "navigate"
url = "/dashboard"

# Use the vision endpoint just for this assertion
[[test.steps]]
kind = "assert"
preset = "no_error_on_page"
endpoint = "vision"
```

**Endpoint types:**

- `llm` — OpenAI-compatible chat completions API. Pricing is per-token
  (`input_per_1m_tokens`, `output_per_1m_tokens`).
- `mcp` — [Model Context Protocol](https://modelcontextprotocol.io) server.
  Launched as a subprocess via `command` + `args`. Pricing is `per_call`.
- `a2a` — [Agent-to-Agent Protocol](https://a2aprotocol.org) agent. Communicates
  via JSON-RPC over HTTP at the given `url`. Pricing is `per_call`.

**Routing:**

- `default_for` lists which task types an endpoint serves automatically
  (`targeting` for element resolution, `assertion` for assertions).
- Add `endpoint = "name"` on any step or `[[test]]` group to override routing.

## A2A agents

Call remote A2A agents in your test scenarios as steps, or use them inside
assertion definitions for reusable agent-backed checks.

### Agent step

```toml
[config.endpoints.audit_bot]
type = "a2a"
url = "http://localhost:9090"
pricing = { per_call = 0.01 }

[[test]]
name = "Audit trail check"
steps = [
    { kind = "navigate", url = "/admin/audit" },
    { kind = "agent", agent = "audit_bot", task = "Check if user 'admin' appears in the recent audit log" },
]
```

### Agent-backed assertions

Define reusable agent assertions with `task_template`:

```toml
[[definitions]]
name = "audit_verify"
agent = "audit_bot"
task_template = "Verify that {expected_text} is true for the page at {url}"

[[test.steps]]
kind = "assert"
definition = "audit_verify"
assert_text = "the user can see the dashboard"
```

Template variables available: `{url}`, `{title}`, `{content}`, `{expected_text}`,
`{description}`, `{task}`.

## Run as an A2A agent

Enable the `a2a-server` feature to expose the framework as an A2A agent that
other agents or orchestrators can call. The server listens on a port and accepts
`tasks/send` JSON-RPC requests.

```toml
[config.a2a_server]
enabled = true
port = 3100
```

```bash
# Build and run with the a2a-server feature
cargo run --features a2a-server -- run scenario.toml --agent-port 3100
```

Or via CLI without modifying the TOML:

```bash
llm-browser-testkit run scenario.toml --agent-port 3100
```

### Docker deployment

```bash
docker build -t llm-browser-testkit .
docker run --rm \
  -e HARNESS_LLM_TEST_URL=https://api.openai.com \
  -e HARNESS_LLM_TEST_MODEL=gpt-4o-mini \
  -e HARNESS_LLM_API_KEY=sk-... \
  llm-browser-testkit run scenario.toml --agent-port 3100 -p 3100:3100
```

A `Dockerfile` is included in the repository — it uses a multi-stage build with
Alpine and Chromium.

## MCP tools

Call MCP server tools directly from test steps to query databases, read files,
or invoke any tool an MCP server exposes.

```toml
[config.endpoints.db]
type = "mcp"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"]
pricing = { per_call = 0.001 }

[[test]]
name = "Database smoke test"
steps = [
    { kind = "navigate", url = "/dashboard" },
    { kind = "mcp", server = "db", tool = "query", args = { sql = "SELECT count(*) FROM users" } },
    { kind = "assert", preset = "no_error_on_page" },
]
```

MCP servers are launched as subprocesses via the configured `command` and `args`.
The framework handles the MCP initialize handshake, tool listing, and invocation
automatically.

## MCP server exposure

Enable the `mcp-server` feature to expose the framework as an MCP server so
other tools can invoke it remotely.

```toml
[config.mcp_server]
enabled = true
port = 3000
```

```bash
cargo run --features mcp-server -- run scenario.toml
```

When enabled, other MCP clients can call tools like `run_scenario` and
`get_page_state` on port 3000.

## Cost tracking & budgets

Every LLM call, agent invocation, and MCP tool call is tracked. After the run
completes, a cost report is printed with per-test and per-endpoint breakdowns.

### Per-test default budget

```toml
[config.budgets.per_test_default]
max_cost = 1.0
max_tokens = 100_000
max_calls = 50
enforcement = "hard"
```

### Per-test override

```toml
[[test]]
name = "Expensive test"
budget = { max_cost = 2.0, max_tokens = 200_000, enforcement = "soft" }
```

### Global budget

```toml
[config.budgets.global]
max_cost = 5.0
max_tokens = 500_000
enforcement = "hard"
```

### Enforcement

| Mode | Behavior |
|------|---------|
| `hard` | Abort the test or run immediately when budget is exceeded |
| `soft` | Print a warning but continue executing remaining steps |

### CLI budgets

```bash
llm-browser-testkit run scenario.toml --max-cost 10.0 --max-tokens 1000000 --budget-enforcement soft
```

### Sample report output

```
═══════════════════════════════════════════════
  COST REPORT
═══════════════════════════════════════════════
  Test: "Homepage loads" — $0.0123 | 1,234 tokens | 4 calls
    endpoint.default:   4 calls,   1,234 tokens, $0.0123
  Test: "Dashboard smoke" — $0.0891 | 4,567 tokens | 6 calls
    endpoint.vision:    2 calls,   3,000 tokens, $0.0450
    endpoint.default:   3 calls,   1,567 tokens, $0.0441
    endpoint.audit_bot:   1 call,   0 tokens, $0.0000
───────────────────────────────────────────────
  GLOBAL SUMMARY
    Total cost:     $0.1014
    Total tokens:   5,801
    Total calls:    10
═══════════════════════════════════════════════
```

## How it works

Four pieces:

1. **Chrome** — launched via the [Chrome DevTools Protocol](https://chromedevtools.github.io/devtools-protocol/)
   (`headless_chrome` crate). It navigates, clicks, types, and extracts page
   content.

2. **LLM** — any OpenAI-compatible API. Used in two places:
   - **Element targeting**: when a step says `target = "the login button"`, the
     runner sends the page's interactive elements to the LLM and asks for a CSS
     selector.
   - **Assertions**: the runner sends page content to the LLM with a QA prompt
     and expects `PASS` or `FAIL: <reason>`.

3. **A2A + MCP** — connect to remote agents via the Agent-to-Agent Protocol
   and to MCP servers for tool-calling. Both are first-class step kinds.

4. **TOML scenarios** — declarative test files. No code, no CSS selectors
   required. Just describe what you want in English.

```
TOML file  →  CLI runner  →  Chrome (CDP)  →  LLM API
                                        →  A2A agent
                                        →  MCP server
```

## Use as a library

```toml
[dependencies]
llm-browser-testkit = { version = "0.1", features = ["macros", "mcp-server"] }
```

```rust
use llm_browser_testkit::runner::ScenarioRunner;
use llm_browser_testkit::scenario::Scenario;

let scenario: Scenario = toml::from_str(&contents)?;
let runner = ScenarioRunner::new(scenario.config.clone(), scenario.definitions);
let report = runner.run(&scenario.test)?;

println!("Passed: {}, Failed: {}", report.tests_passed, report.tests_failed);

// Access cost/usage data
let usage = runner.usage_tracker();
let global = usage.global_snapshot();
println!("Total cost: ${:.4}", global.total_cost);

// Print the cost report
llm_browser_testkit::reporting::print_report(
    &usage.per_test_snapshots(),
    &global,
);
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

## LLM authentication

The runner supports API keys and custom headers for SSO or alternative auth:

```toml
[config]
llm_api_key = "sk-..."
llm_headers = { "X-Org-ID" = "acme", "X-Project" = "qa" }
```

Endpoints can also carry their own credentials:

```toml
[config.endpoints.production]
type = "llm"
url = "https://api.openai.com"
api_key = "sk-prod-..."
model = "gpt-4o"
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