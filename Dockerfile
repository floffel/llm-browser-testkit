# Build stage
FROM rust:1.89-alpine AS builder

RUN apk add --no-cache musl-dev gcc make

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY tests/ tests/
COPY examples/ examples/

RUN cargo build --release --all-features

# Runtime stage
FROM alpine:3.23.5

ARG ENABLE_AWS_CLI=false
ARG ENABLE_AZURE_CLI=false

RUN apk add --no-cache chromium chromium-chromedriver ca-certificates

RUN if [ "$ENABLE_AWS_CLI" = "true" ]; then \
        apk add --no-cache aws-cli; \
    fi

RUN if [ "$ENABLE_AZURE_CLI" = "true" ]; then \
        apk add --no-cache python3 py3-pip \
        && pip install --no-cache-dir --break-system-packages azure-cli; \
    fi

ENV CHROME_BIN=/usr/bin/chromium-browser
ENV CHROMEDRIVER=/usr/bin/chromedriver

# Keep the runtime PATH explicit so the llm-browser-testkit binary (and the
# entrypoint that resolves it) stays reachable even if the base image's
# default PATH ever changes.
ENV PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

COPY --from=builder /app/target/release/llm-browser-testkit /usr/local/bin/llm-browser-testkit
COPY default-scenario.toml /default-scenario.toml
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 3100
# Entrypoint dispatcher: stays alive when used as a job container
# (act_runner/GitHub Actions `container.image`), prints the version, and
# forwards args for one-shot CLI use (`docker run <image> run scenario.toml`).
# No CMD on purpose: an empty command lets the entrypoint tail when booted
# as a job container; explicitly passed args (e.g. `run ...`) reach the
# harness untouched.
ENTRYPOINT ["/entrypoint.sh"]