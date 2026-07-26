# Build stage
FROM rust:1.89-alpine AS builder

RUN apk add --no-cache musl-dev openssl-dev openssl-libs-static

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY examples/ examples/

RUN cargo build --release --all-features

# Runtime stage
FROM alpine:3.21

RUN apk add --no-cache chromium chromium-chromedriver ca-certificates

ENV CHROME_BIN=/usr/bin/chromium-browser
ENV CHROMEDRIVER=/usr/bin/chromedriver

COPY --from=builder /app/target/release/llm-browser-testkit /usr/local/bin/llm-browser-testkit

ENTRYPOINT ["llm-browser-testkit"]