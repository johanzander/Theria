ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base-python:3.11
FROM $BUILD_FROM

# Set version labels
ARG BUILD_VERSION=0.1.0
ARG BUILD_DATE
ARG BUILD_REF

# Labels
LABEL \
    io.hass.name="Theria" \
    io.hass.description="Smart heating optimization using electricity price-based heat capacitor strategy" \
    io.hass.version=${BUILD_VERSION} \
    io.hass.type="addon" \
    io.hass.arch="aarch64,amd64,armhf,armv7,i386" \
    maintainer="Johan Zander <johanzander@gmail.com>" \
    org.label-schema.build-date=${BUILD_DATE} \
    org.label-schema.description="Smart heating optimization using electricity price-based heat capacitor strategy" \
    org.label-schema.name="Theria" \
    org.label-schema.schema-version="1.0" \
    org.label-schema.vcs-ref=${BUILD_REF}

# Install requirements
RUN apk add --no-cache \
    python3 \
    py3-pip \
    python3-dev \
    gcc \
    musl-dev \
    bash

# Set working directory
WORKDIR /app

# Copy Python application files from backend directory
COPY backend/app.py backend/api.py backend/log_config.py backend/requirements.txt ./

# Copy core directory
COPY core/ /app/core/

# Copy run script
COPY backend/run.sh ./

# Create and use virtual environment
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Install Python requirements in the virtual environment
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Make scripts executable
RUN chmod a+x /app/run.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8081/api/health', timeout=5)" || exit 1

# Expose the port
EXPOSE 8081

# Launch application
CMD ["/bin/bash", "/app/run.sh"]
