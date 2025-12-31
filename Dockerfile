ARG RUST_VERSION=1.92.0
ARG APP_NAME=pdf_to_webp

FROM rust:${RUST_VERSION}-trixie AS builder


# Build Stage

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY akaze ./akaze
COPY src ./src
COPY libpdfium.so ./libpdfium.so
RUN cargo build --release


# Bundle Stage
FROM  debian:trixie-slim
WORKDIR /app
COPY --from=builder /app/target/release/pdf_to_webp .
COPY --from=builder /app/libpdfium.so .
USER 1000
CMD ["./pdf_to_webp", "--mq"]





