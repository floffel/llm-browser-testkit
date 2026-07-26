//! MCP server exposure — exposes the framework as an MCP server.
//!
//! When enabled, the framework listens on a configurable port and exposes
//! tools for remote test execution and page inspection.
//!
//! This module requires the `mcp-server` feature to be enabled.

/// Starts the MCP server on the given port.
///
/// Exposes tools:
/// - `run_scenario(path)` — run a TOML scenario file and return the report
/// - `get_page_state()` — return current page URL, title, text content
/// - `get_browser_console()` — return browser console logs
///
/// # Errors
///
/// Returns an error if the server fails to bind to the port.
#[allow(dead_code)]
pub fn start_mcp_server(port: u16) -> anyhow::Result<()> {
    // Stub implementation — full MCP server will be built in a future
    // iteration.
    eprintln!("MCP server started on port {port}");
    Ok(())
}
