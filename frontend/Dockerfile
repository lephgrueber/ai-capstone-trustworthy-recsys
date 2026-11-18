# syntax=docker/dockerfile:1.7

ARG NODE_IMAGE=node:24.11.1-alpine3.22
ARG NGINX_IMAGE=nginxinc/nginx-unprivileged:1.29.4-alpine

FROM ${NODE_IMAGE} AS build
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM ${NGINX_IMAGE} AS runtime
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /frontend/dist/ /usr/share/nginx/html/
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=12 \
    CMD wget --quiet --tries=1 --spider http://127.0.0.1:8080/ready || exit 1
