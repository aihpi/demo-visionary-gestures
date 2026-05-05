# Stage 1: Build React/Vite app
# Pre-converted TF.js model files are in public/model/ (committed to repo)
FROM node:22-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build -- --base=/

# Stage 2: Serve with nginx
FROM nginx:alpine
RUN sed -i 's/worker_processes\s*auto;/worker_processes 1;/' /etc/nginx/nginx.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
