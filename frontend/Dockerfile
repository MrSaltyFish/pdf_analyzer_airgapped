# Base image
FROM node:20-alpine

# Working directory
WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy source
COPY . .

# Build (optional for dev)
# RUN npm run build

EXPOSE 3000

CMD ["npm", "run", "dev"]
