FROM node:20-alpine
LABEL Name=printperfect Version=1.0.0

WORKDIR /app

# Copy all workspace files first (cleaner approach for spaces in directory names)
COPY . .

# Install dependencies
RUN cd "Print Perfet" && npm install --omit=dev

# Set working directory to app code
WORKDIR /app/Print Perfet

# Expose port
EXPOSE 3000

# Run production server
ENV NODE_ENV=production
CMD ["npm", "start"]
