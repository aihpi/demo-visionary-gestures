# Stage 1: Convert TF SavedModel → TF.js Graph Model format
FROM python:3.11-slim AS model-convert
WORKDIR /model
COPY model/ ./
RUN pip install --no-cache-dir tensorflowjs
RUN tensorflowjs_converter \
    --input_format=tf_saved_model \
    rps_model_savedmodel/ \
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
