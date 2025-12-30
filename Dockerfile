ARG RUST_VERSION=1.92.0
ARG APP_NAME=pdf_to_webp

FROM rust:${RUST_VERSION}-alpine AS builder
ARG APP_NAME
WORKDIR /usr/src/


# Build Stage
RUN rustup target add x86_64-unknown-linux-musl

RUN USER=root cargo new ${APP_NAME}
WORKDIR /usr/src/${APP_NAME}
COPY Cargo.toml Cargo.lock ./
COPY akaze/Cargo.toml akaze/Cargo.lock ./akaze/
COPY akaze ./akaze
COPY src ./src

RUN cargo build --release

RUN cargo install --target x86_64-unknown-linux-musl --path .

# Bundle Stage
FROM scratch
COPY --from=builder /usr/local/cargo/bin/pdf_to_webp .
USER 1000
CMD ["./pdf_to_webp", "--mq"]





