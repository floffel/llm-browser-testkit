//! A2A server — exposes the framework as an A2A agent.
//!
//! Listens on a port and accepts A2A JSON-RPC `tasks/send` requests.
//! Each task is treated as a scenario execution request.
//!
//! Requires the `a2a-server` feature to be enabled.

use std::net::SocketAddr;
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

/// Shared state for the A2A server.
#[derive(Default)]
pub struct A2aServerState {
    /// Accumulated results from executed tasks.
    pub results: Vec<String>,
}

/// Starts an A2A server on the given port.
///
/// Accepts `tasks/send` JSON-RPC requests and responds with a dummy
/// result. In production, this would execute TOML scenarios.
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
        .with_context(|| format!("A2A server: failed to bind to port {port}"))?;

    let state = Arc::new(Mutex::new(A2aServerState::default()));

    eprintln!("A2A agent server listening on port {port}");

    loop {
        let (stream, _) = listener
            .accept()
            .await
            .context("A2A server: accept failed")?;
        let io = TokioIo::new(stream);
        let state = Arc::clone(&state);

        tokio::spawn(async move {
            let svc = service_fn(move |req| handle_a2a_request(req, Arc::clone(&state)));
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

            // Record the task
            {
                let mut st = state.lock().await;
                st.results.push(format!("task received: {task}"));
            }

            let response = serde_json::json!({
                "jsonrpc": "2.0",
                "result": {
                    "id": "task-1",
                    "messages": [{
                        "parts": [{
                            "type": "text",
                            "text": format!("Acknowledged: {task}")
                        }]
                    }]
                },
                "id": id
            });

            Ok(Response::builder()
                .status(200)
                .header("Content-Type", "application/json")
                .body(serde_json::to_string(&response).unwrap())
                .unwrap())
        }
        _ => Ok(error_response(
            -32601,
            &format!("Method not found: {method}"),
            Some(id),
        )),
    }
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
}
