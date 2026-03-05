#!/bin/bash
# Wir entfernen -u, um Fehler bei leeren Variablen zu vermeiden
set -e 

echo "--- Initialisiere VM-System ---"

# Prüfe CPU-Virtualisierung
if [ -e /dev/kvm ]; then
    echo "KVM-Beschleunigung gefunden."
else
    # WICHTIG: Das ist kritisch für die Performance deiner Masterarbeit!
    echo "Warnung: KVM (Hardware-Virtualisierung) nicht gefunden. Die VM wird sehr langsam sein."
fi

# Sicherer Zugriff auf die Variable CPU
# Falls $CPU nicht existiert, wird "max" als Standard genutzt
CPU_MODEL="${CPU:-max}"

echo "CPU-Modell gesetzt auf: $CPU_MODEL"

# Verzeichnisse vorbereiten
mkdir -p /run/conf
mkdir -p /run/assets
mkdir -p /run/shm
touch /run/shm/status.html

echo "Initialisierung abgeschlossen."