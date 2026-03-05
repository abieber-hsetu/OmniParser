# Definiere die Architektur-Argumente für Multi-Arch-Builds
ARG TARGETARCH
ARG VERSION_ARG="latest"

# --- AMD64 STAGE ---
FROM scratch AS build-amd64
# Nutze das stabile qemux Image
COPY --from=qemux/qemu:latest / /

ARG DEBCONF_NOWARNINGS="yes"
ARG DEBIAN_FRONTEND="noninteractive"
ARG DEBCONF_NONINTERACTIVE_SEEN="true"

RUN set -eu && \
    apt-get update && \
    apt-get --no-install-recommends -y install \
        bc \
        jq \
        curl \
        7zip \
        python3 \
        samba \
        xz-utils \
        wimtools \
        dos2unix \
        cabextract \
        genisoimage \
        libxml2-utils \
        libarchive-tools \
        novnc \
        python3-websockify && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Kopiere Skripte und Assets
COPY --chmod=755 ./vm/buildcontainer /run/

# 2. Skripte reparieren (Installieren -> Ausführen -> Aufräumen)
RUN apt-get update && apt-get install -y dos2unix && \
    find /run -type f -name "*.sh" -exec dos2unix {} + && \
    apt-get purge -y dos2unix && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# 3. Assets kopieren (falls vorhanden)
COPY --chmod=755 ./vm/win11def /run/assets
RUN [ -d /run/assets ] && find /run/assets -type f -exec dos2unix {} + || true

RUN apt-get update && apt-get install -y dos2unix && \
    find /run -type f -name "*.sh" -exec dos2unix {} +

COPY ./vm/buildcontainer/network.sh /run/network.sh
COPY ./vm/buildcontainer/define.sh /run/define.sh

# 2. Sicherstellen, dass die Dateien ausführbar sind und KEINE Windows-Zeilenenden haben
RUN chmod +x /run/network.sh && \
    chmod +x /run/define.sh && \
    dos2unix /run/network.sh && \
    dos2unix /run/define.sh

# Manuelle Installation von WSDD (da kein Apt-Paket vorhanden)
ADD --chmod=755 https://raw.githubusercontent.com/christgau/wsdd/v0.8/src/wsdd.py /usr/sbin/wsdd
# Treiber herunterladen
ADD --chmod=664 https://github.com/qemus/virtiso-whql/releases/download/v1.9.43-0/virtio-win-1.9.43.tar.xz /drivers.txz

# --- ARM64 STAGE ---
FROM dockurr/windows-arm:${VERSION_ARG} AS build-arm64

# --- FINAL STAGE ---
# Hier wird automatisch zwischen build-amd64 und build-arm64 gewählt
FROM build-${TARGETARCH}

ARG VERSION_ARG="0.00"
RUN echo "$VERSION_ARG" > /run/version

# Ports für Web-Interface (8006) und RDP (3389)
EXPOSE 8006 3389

ENV VERSION="win11e"

# Startbefehl via Tini-Init
ENTRYPOINT ["/usr/bin/tini", "-s", "/run/entry.sh"]