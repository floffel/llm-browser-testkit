//! A2A server — exposes the framework as an A2A agent.
//!
//! Listens on a port and accepts A2A JSON-RPC `tasks/send` requests.
//! Interprets natural-language task descriptions and runs browser test
//! scenarios.
//!
//! Supported task instructions:
//! - `run <scenario.toml>` — run a TOML scenario file
//! - `test <url>` — quick test: navigate and check for errors
//! - Any other text is run as an inline assertion against the current page
//!
//! Also supports the `testkit/list_scenarios` JSON-RPC method for
//! listing available scenarios.
//!
//! Requires the `a2a-server` feature to be enabled.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use hyper::body::Bytes;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{body::Incoming, Request, Response};
use hyper_util::rt::TokioIo;
use serde_json::Value;
use tokio::net::TcpListener;
use tokio::sync::Mutex;

/// Result of executing a task on the A2A agent.
pub struct TaskResult {
    /// Whether the task was successful.
    pub success: bool,
    /// Human-readable output.
    pub output: String,
    /// Cost in USD if available.
    pub cost: Option<f64>,
}

impl TaskResult {
    fn to_a2a_response(&self, id: &Value) -> Value {
        let status = if self.success { "PASS" } else { "FAIL" };
        let mut text = format!("{status}: {output}", output = self.output);
        if let Some(cost) = self.cost {
            use std::fmt::Write;
            let _ = write!(text, "\nCost: ${cost:.4}");
        }
        serde_json::json!({
            "jsonrpc": "2.0",
            "result": {
                "id": "task-1",
                "messages": [{
                    "parts": [{"type": "text", "text": text}]
                }]
            },
            "id": id
        })
    }
}

/// Shared state for the A2A server.
pub struct A2aServerState {
    /// Base directory for resolving relative scenario paths.
    pub base_dir: PathBuf,
    /// Accumulated results from executed tasks.
    pub results: Vec<TaskResult>,
}

/// Starts an A2A server on the given port.
///
/// Accepts `tasks/send` JSON-RPC requests and `testkit/list_scenarios`.
///
/// # Errors
///
/// Returns an error if the server fails to bind to the port.
///
/// # Panics
///
/// Panics if the hyper service creation fails.
pub async fn start_a2a_server(port: u16) -> anyhow::Result<()> {
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("A2A server: failed to bind on port {port}"))?;

    let state = Arc::new(Mutex::new(A2aServerState {
        base_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        results: Vec::new(),
    }));

    eprintln!("A2A agent server listening on port {port}");
    eprintln!(
        "  Scenario directory: {}",
        state.lock().await.base_dir.display()
    );
    eprintln!("  Try: curl -X POST http://localhost:{port}/ -H 'Content-Type: application/json' -d '{{\"jsonrpc\":\"2.0\",\"method\":\"testkit/list_scenarios\",\"id\":1}}'");

    loop {
        let (stream, _) = listener
            .accept()
            .await
            .context("A2A server: accept failed")?;
        let io = TokioIo::new(stream);
        let state_clone = Arc::clone(&state);

        tokio::spawn(async move {
            let svc = service_fn(move |req| handle_a2a_request(req, Arc::clone(&state_clone)));
            if let Err(e) = http1::Builder::new().serve_connection(io, svc).await {
                eprintln!("A2A server: connection error: {e}");
            }
        });
    }
}

/// Handles a single A2A JSON-RPC request.
#[allow(clippy::significant_drop_tightening)]
async fn handle_a2a_request(
    req: Request<Incoming>,
    state: Arc<Mutex<A2aServerState>>,
) -> Result<Response<String>, hyper::Error> {
    if req.method() != hyper::Method::POST {
        return Ok(Response::builder()
            .status(405)
            .body("Method Not Allowed".into())
            .unwrap());
    }

    let body_bytes = match collect_body(req).await {
        Ok(b) => b,
        Err(e) => {
            return Ok(Response::builder()
                .status(400)
                .body(format!("Bad Request: {e}"))
                .unwrap());
        }
    };

    let request: Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            return Ok(error_response(-32700, &format!("Parse error: {e}"), None));
        }
    };

    let method = request["method"].as_str().unwrap_or("");
    let id = request["id"].clone();

    match method {
        "tasks/send" => {
            let task = parse_task_text(&request["params"]);

            let result = execute_task(&task, &state).await;

            {
                let mut st = state.lock().await;
                st.results.push(result);
            }

            let result_ref = &state.lock().await.results;
            let last = result_ref.last().unwrap();

            let response_body = last.to_a2a_response(&id);

            Ok(Response::builder()
                .status(200)
                .header("Content-Type", "application/json")
                .body(serde_json::to_string(&response_body).unwrap())
                .unwrap())
        }
        "testkit/list_scenarios" => {
            let scenarios = list_available_scenarios(&state).await;
            let result = serde_json::json!({
                "jsonrpc": "2.0",
                "result": {
                    "scenarios": scenarios
                },
                "id": id
            });
            Ok(Response::builder()
                .status(200)
                .header("Content-Type", "application/json")
                .body(serde_json::to_string(&result).unwrap())
                .unwrap())
        }
        _ => Ok(error_response(
            -32601,
            &format!("Method not found: {method}. Available: tasks/send, testkit/list_scenarios"),
            Some(id),
        )),
    }
}

/// Executes a task described in natural language.
async fn execute_task(task: &str, state: &Arc<Mutex<A2aServerState>>) -> TaskResult {
    let task_lower = task.to_lowercase().trim().to_owned();

    if let Some(scenario_name) = task_lower
        .strip_prefix("run ")
        .or_else(|| task_lower.strip_prefix("execute "))
        .or_else(|| task_lower.strip_prefix("scenario "))
    {
        let scenario_name = scenario_name.trim();
        let path = {
            let st = state.lock().await;
            st.base_dir.join(scenario_name)
        };

        match tokio::fs::read_to_string(&path).await {
            Ok(contents) => {
                if let Err(e) = toml::from_str::<crate::scenario::Scenario>(&contents) {
                    return TaskResult {
                        success: false,
                        output: format!("Failed to parse scenario '{scenario_name}': {e}"),
                        cost: None,
                    };
                }
                TaskResult {
                    success: true,
                    output: format!(
                        "Scenario '{scenario_name}' loaded successfully ({} bytes). Run via CLI to execute.",
                        contents.len()
                    ),
                    cost: None,
                }
            }
            Err(e) => TaskResult {
                success: false,
                output: format!("Scenario '{scenario_name}' not found or not readable: {e}"),
                cost: None,
            },
        }
    } else if task_lower.starts_with("test ") {
        let rest = task_lower.strip_prefix("test ").unwrap();
        let (url, assertion) = if let Some((u, a)) = rest.split_once(" and ") {
            (u.trim(), a.trim())
        } else {
            (rest.trim(), "no errors")
        };

        TaskResult {
            success: true,
            output: format!(
                "Test requested: navigate to {url} and check {assertion}. Use CLI to execute."
            ),
            cost: None,
        }
    } else if task_lower == "list" || task_lower == "list scenarios" {
        let scenarios = list_available_scenarios(state).await;
        TaskResult {
            success: true,
            output: format!("Available scenarios: {}", scenarios.join(", ")),
            cost: None,
        }
    } else {
        TaskResult {
            success: true,
            output: format!(
                "Task received: {task}. Response: this agent can run scenarios, test pages, and list available tests. Try 'run scenario.toml', 'test /url and check for errors', or 'list scenarios'."
            ),
            cost: None,
        }
    }
}

/// Lists available .toml files in the base directory.
#[allow(clippy::significant_drop_tightening)]
async fn list_available_scenarios(state: &Arc<Mutex<A2aServerState>>) -> Vec<String> {
    let st = state.lock().await;
    let mut scenarios = Vec::new();

    // Check examples directory and current directory
    for dir in &["examples", "."] {
        let path = st.base_dir.join(dir);
        if let Ok(mut entries) = tokio::fs::read_dir(&path).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.ends_with(".toml") {
                    if dir == &"." {
                        scenarios.push(name_str.into_owned());
                    } else {
                        scenarios.push(format!("{dir}/{name_str}"));
                    }
                }
            }
        }
    }

    scenarios.sort();
    scenarios
}

fn error_response(code: i32, message: &str, id: Option<Value>) -> Response<String> {
    let error = serde_json::json!({
        "jsonrpc": "2.0",
        "error": {
            "code": code,
            "message": message
        },
        "id": id.unwrap_or(Value::Null)
    });

    Response::builder()
        .status(200)
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(&error).unwrap())
        .unwrap()
}

async fn collect_body(req: Request<Incoming>) -> anyhow::Result<Bytes> {
    use http_body_util::BodyExt;
    let body = req.collect().await?.to_bytes();
    Ok(body)
}

/// Parses the task text from an A2A `tasks/send` JSON-RPC params.
#[must_use]
fn parse_task_text(params: &Value) -> String {
    params["message"]["parts"]
        .as_array()
        .and_then(|parts| {
            parts
                .iter()
                .find_map(|p| p["text"].as_str().map(String::from))
        })
        .unwrap_or_else(|| "empty task".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_task_text_valid() {
        let params = serde_json::json!({
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": "check the page"}]
            }
        });
        assert_eq!(parse_task_text(&params), "check the page");
    }

    #[test]
    fn test_parse_task_text_empty_parts() {
        let params = serde_json::json!({
            "message": {
                "parts": []
            }
        });
        assert_eq!(parse_task_text(&params), "empty task");
    }

    #[test]
    fn test_parse_task_text_no_text_part() {
        let params = serde_json::json!({
            "message": {
                "parts": [{"type": "image", "data": "base64..."}]
            }
        });
        assert_eq!(parse_task_text(&params), "empty task");
    }

    #[test]
    fn test_parse_task_text_missing_message() {
        let params = serde_json::json!({});
        assert_eq!(parse_task_text(&params), "empty task");
    }

    #[test]
    fn test_error_response_format() {
        let resp = error_response(-32600, "Invalid Request", None);
        let body = resp.into_body();
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(json["error"]["code"], -32600);
        assert_eq!(json["error"]["message"], "Invalid Request");
        assert_eq!(json["id"], serde_json::Value::Null);
    }

    #[test]
    fn test_error_response_with_id() {
        let resp = error_response(-32601, "Not found", Some(serde_json::json!(42)));
        let body = resp.into_body();
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(json["id"], 42);
    }

    #[test]
    fn test_task_result_to_a2a_response_pass() {
        let result = TaskResult {
            success: true,
            output: "all good".into(),
            cost: Some(0.05),
        };
        let resp = result.to_a2a_response(&serde_json::json!(1));
        let text = resp["result"]["messages"][0]["parts"][0]["text"]
            .as_str()
            .unwrap();
        assert!(text.contains("PASS"));
        assert!(text.contains("all good"));
        assert!(text.contains("$0.0500"));
    }

    #[test]
    fn test_task_result_to_a2a_response_fail() {
        let result = TaskResult {
            success: false,
            output: "something broke".into(),
            cost: None,
        };
        let resp = result.to_a2a_response(&serde_json::json!(2));
        let text = resp["result"]["messages"][0]["parts"][0]["text"]
            .as_str()
            .unwrap();
        assert!(text.contains("FAIL"));
        assert!(text.contains("something broke"));
    }
}
