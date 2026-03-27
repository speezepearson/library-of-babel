FROM node:22-bookworm

RUN apt-get update && \
    apt-get install -y \
        git \
        curl \
        sudo \
        jq \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash -u 60000 dev && \
    echo 'dev ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/dev && \
    chmod 0440 /etc/sudoers.d/dev && \
    passwd -d root
USER dev

WORKDIR /home/dev

ENV PATH="/home/dev/.local/bin:$PATH"
RUN curl -fsSL https://claude.ai/install.sh | bash

COPY .disposables/claude.json /home/dev/.claude.json

WORKDIR /tmp/preinstalls
COPY package.json package-lock.json* ./
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi
