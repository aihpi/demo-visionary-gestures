# Stage 1: Convert Keras H5 model → TF.js Layers format
# Pin TF to 2.15.x to avoid protobuf>=6 conflict introduced by tensorflow_decision_forests in TF 2.16+
FROM python:3.11-slim AS model-convert
WORKDIR /model
COPY model/ ./
RUN pip install --no-cache-dir \
    "tensorflow==2.15.0" \
    "tensorflowjs==4.20.0"
RUN tensorflowjs_converter \
    --input_format=keras \
    rps_model.h5 \
    /tfjs_model/

# Stage 2: Build React/Vite app
FROM node:22-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN mkdir -p public/model
COPY --from=model-convert /tfjs_model/ ./public/model/
RUN npm run build -- --base=/

# Stage 3: Serve with nginx
FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
