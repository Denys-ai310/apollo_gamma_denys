# Use an official Python runtime as the base image
FROM python:3.9.11

# Set working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    fonts-liberation \
    wget \
    && wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /usr/share/keyrings/microsoft.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/microsoft.gpg] https://packages.microsoft.com/debian/11/prod bullseye main" > /etc/apt/sources.list.d/microsoft.list \
    && apt-get update \
    && apt-get install -y ttf-mscorefonts-installer \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file (you'll need to create this)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files into the container
COPY . .

# Make the training scripts executable
RUN chmod +x training/btc/24h/*.sh

# Set the default command
CMD ["bash", "training/btc/24h/train_all_sequentially.sh"]