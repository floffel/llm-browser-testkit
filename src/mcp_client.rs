//! MCP (Model Context Protocol) client.
//!
//! Connects to MCP servers via stdio (subprocess) or HTTP transport,
//! lists tools, and calls them.

use std::io::BufRead;
use std::io::Write;
use std::process::Child;
use std::process::Command;
use std::process::Stdio;
use std::time::Duration;

use anyhow::Context;
use serde_json::Value;

/// Transport mode for an MCP connection.
#[derive(Debug)]
pub enum McpTransport {
    /// Spawn a subprocess and communicate via stdio (stdin/stdout).
    Stdio(Child),
    /// HTTP-based transport (streamable HTTP or SSE).
    #[allow(dead_code)]
    Http {
        /// Server base URL for HTTP requests.
        url: String,
        /// HTTP client instance.
        client: reqwest::Client,
    },
}

/// MCP client for a connected server.
#[derive(Debug)]
pub struct McpClient {
    transport: McpTransport,
    /// Cached request ID counter.
    next_id: u64,
}

impl McpClient {
    /// Connects to an MCP server via stdio, spawning the given command.
    ///
    /// Sends the `initialize` request and waits for the response.
    ///
    /// # Errors
    ///
    /// Returns an error if the subprocess cannot be spawned, the initialize
    /// handshake fails, or the server is unreachable.
    pub fn connect_stdio(command: &str, args: &[String]) -> anyhow::Result<Self> {
        let child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("MCP: failed to spawn server process")?;

        let mut client = Self {
            transport: McpTransport::Stdio(child),
            next_id: 1,
        };

        // MCP initialize handshake
        let init = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "llm-browser-testkit",
                    "version": "0.1.2"
                }
            },
            "id": 0
        });
        let _response = client.send_request(&init)?;

        // Send initialized notification
        let initialized = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        });
        client.send_request(&initialized)?;

        Ok(client)
    }

    /// Lists available tools on the MCP server.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails or the server returns an error.
    pub fn list_tools(&mut self) -> anyhow::Result<Vec<McpTool>> {
        let req = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": 0
        });
        let resp = self.send_request(&req)?;
        let tools: Vec<McpTool> = serde_json::from_value(resp["result"]["tools"].clone())
            .context("MCP: failed to parse tools list")?;
        Ok(tools)
    }

    /// Calls a tool on the MCP server with the given arguments.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails or the server returns an error.
    pub fn call_tool(&mut self, tool_name: &str, args: &Value) -> anyhow::Result<McpToolResult> {
        let params = if args.is_null() {
            serde_json::json!({ "name": tool_name })
        } else {
            serde_json::json!({
                "name": tool_name,
                "arguments": args
            })
        };

        let req = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": params,
            "id": 0
        });
        let resp = self.send_request(&req)?;
        let result: McpToolResult = serde_json::from_value(resp["result"].clone())
            .context("MCP: failed to parse tool result")?;
        Ok(result)
    }

    fn send_request(&mut self, request: &Value) -> anyhow::Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        match &mut self.transport {
            McpTransport::Stdio(ref mut child) => send_request_stdio(id, child, request),
            McpTransport::Http { .. } => {
                anyhow::bail!("MCP HTTP transport not yet implemented")
            }
        }
    }
}

/// Sends a JSON-RPC request over stdio to an MCP server child process.
#[allow(clippy::significant_drop_tightening)]
fn send_request_stdio(_id: u64, child: &mut Child, request: &Value) -> anyhow::Result<Value> {
    let mut request_str = serde_json::to_string(request)?;
    request_str.push('\n');

    let stdin = child.stdin.as_mut().context("MCP: stdin not available")?;
    stdin
        .write_all(request_str.as_bytes())
        .context("MCP: write to stdin failed")?;
    stdin.flush().context("MCP: flush stdin failed")?;

    let stdout = child.stdout.as_mut().context("MCP: stdout not available")?;
    let mut reader = std::io::BufReader::new(stdout);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .context("MCP: read from stdout failed")?;

    let resp: Value = serde_json::from_str(&line).context("MCP: failed to parse JSON response")?;

    if let Some(error) = resp["error"]["message"].as_str() {
        anyhow::bail!("MCP error: {error}");
    }

    Ok(resp)
}

// Suppress dead_code on connect_http for now
#[allow(dead_code)]
impl McpClient {
    /// Connects to an MCP server via HTTP transport.
    ///
    /// # Errors
    ///
    /// Returns an error if the server is unreachable or the initialize
    /// handshake fails.
    #[allow(dead_code)]
    pub async fn connect_http(url: &str, timeout: Duration) -> anyhow::Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .context("build reqwest client")?;

        let mcp = Self {
            transport: McpTransport::Http {
                url: url.trim_end_matches('/').to_owned(),
                client: client.clone(),
            },
            next_id: 1,
        };

        // Initialize handshake over HTTP
        let init = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "llm-browser-testkit",
                    "version": "0.1.2"
                }
            },
            "id": 0
        });

        let resp = client
            .post(url)
            .header("Content-Type", "application/json")
            .json(&init)
            .send()
            .await
            .context("MCP HTTP: initialize failed")?;

        let json: Value = resp.json().await.context("MCP HTTP: parse response")?;
        if let Some(error) = json["error"]["message"].as_str() {
            anyhow::bail!("MCP error: {error}");
        }

        Ok(mcp)
    }
}

/// A tool exposed by an MCP server.
#[derive(Debug, Clone, serde::Deserialize)]
#[allow(non_snake_case)]
pub struct McpTool {
    /// Tool name.
    pub name: String,
    /// Human-readable description.
    #[serde(default)]
    pub description: String,
    /// JSON Schema for tool arguments.
    #[serde(default)]
    pub inputSchema: Value,
}

/// Result from calling an MCP tool.
#[derive(Debug, Clone, serde::Deserialize)]
#[allow(non_snake_case)]
pub struct McpToolResult {
    /// Content blocks returned by the tool.
    #[serde(default)]
    pub content: Vec<McpContent>,
    /// Whether the result is an error.
    #[serde(default)]
    pub isError: bool,
}

impl std::fmt::Display for McpToolResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, c) in self.content.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            match c {
                McpContent::Text { text } => write!(f, "{text}")?,
                McpContent::Resource { resource } => {
                    write!(f, "[resource: {}]", resource.uri)?;
                }
                McpContent::Image { data, mimeType } => {
                    write!(f, "[image: {mimeType}, {} bytes]", data.len())?;
                }
            }
        }
        Ok(())
    }
}

/// Content block within an MCP tool result.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(tag = "type")]
#[allow(non_snake_case)]
pub enum McpContent {
    /// Plain text content.
    #[serde(rename = "text")]
    Text {
        /// The text content.
        text: String,
    },
    /// A resource reference.
    #[serde(rename = "resource")]
    Resource {
        /// The resource descriptor.
        resource: McpResource,
    },
    /// Base64-encoded image.
    #[serde(rename = "image")]
    Image {
        /// Base64-encoded image data.
        data: String,
        /// MIME type (e.g. image/png).
        mimeType: String,
    },
}

/// Resource descriptor from an MCP response.
#[derive(Debug, Clone, serde::Deserialize)]
#[allow(non_snake_case)]
pub struct McpResource {
    /// Resource URI.
    pub uri: String,
    /// MIME type.
    #[serde(default)]
    pub mimeType: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_tool_result_display_text() {
        let result = McpToolResult {
            content: vec![McpContent::Text {
                text: "hello world".into(),
            }],
            isError: false,
        };
        assert_eq!(result.to_string(), "hello world");
    }

    #[test]
    fn test_mcp_tool_result_display_multiple() {
        let result = McpToolResult {
            content: vec![
                McpContent::Text {
                    text: "line1".into(),
                },
                McpContent::Text {
                    text: "line2".into(),
                },
            ],
            isError: false,
        };
        assert_eq!(result.to_string(), "line1\nline2");
    }

    #[test]
    fn test_mcp_tool_result_display_resource() {
        let result = McpToolResult {
            content: vec![McpContent::Resource {
                resource: McpResource {
                    uri: "file:///test".into(),
                    mimeType: "text/plain".into(),
                },
            }],
            isError: false,
        };
        assert_eq!(result.to_string(), "[resource: file:///test]");
    }

    #[test]
    fn test_mcp_tool_result_display_image() {
        let result = McpToolResult {
            content: vec![McpContent::Image {
                data: "base64data".into(),
                mimeType: "image/png".into(),
            }],
            isError: false,
        };
        assert_eq!(result.to_string(), "[image: image/png, 10 bytes]");
    }

    #[test]
    fn test_mcp_tool_deserialize() {
        let json = serde_json::json!({
            "name": "query",
            "description": "Run a SQL query",
            "inputSchema": {"type": "object"}
        });
        let tool: McpTool = serde_json::from_value(json).unwrap();
        assert_eq!(tool.name, "query");
        assert_eq!(tool.description, "Run a SQL query");
    }
}
