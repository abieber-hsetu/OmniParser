#!/usr/bin/env bash
set -Eeuo pipefail

# Hilfsfunktionen, die oft aufgerufen werden (Dummys)
info() { echo "[INFO] $1"; }
warn() { echo "[WARN] $1"; }
error() { echo "[ERROR] $1"; exit 1; }
html() { echo "<p>$1</p>" >> /run/shm/status.html; }

# Verhindert alle "unbound variable" Fehler auf einmal
: "${DEBUG:="N"}"; : "${PLATFORM:="x64"}"; : "${ENGINE:="docker"}"; 
: "${RAM_SIZE:="4G"}"; : "${CPU_CORES:="2"}"; : "${DISK_SIZE:="64G"}";
: "${BOOT_INDEX:="9"}"; : "${BOOT_MODE:="uefi"}"; : "${VNC_PORT:="5900"}";
: "${WSS_PORT:="8001"}"; : "${WEB_PORT:="8006"}"; : "${MON_PORT:="8000"}";
: "${DHCP:="N"}"; : "${BRIDGE:="br0"}"; : "${PROCESS:="qemu"}";
: "${HOST:="windows"}"; : "${KERNEL:="5"}"; : "${WSD_PORT:="3702"}";
: "${ALLOCATE:="N"}"; : "${STORAGE:="/storage"}"; : "${APP:="windows"}";
export DEBUG PLATFORM ENGINE RAM_SIZE CPU_CORES DISK_SIZE BOOT_INDEX BOOT_MODE VNC_PORT WSS_PORT WEB_PORT MON_PORT DHCP BRIDGE PROCESS HOST KERNEL WSD_PORT ALLOCATE STORAGE APP

# --- 1. SYSTEM & IDENTITÄT ---
: "${APP:="windows"}"
: "${VERSION:="11"}"
: "${ENGINE:="docker"}"
: "${ROOTLESS:="N"}"
: "${PRIVILEGED:="N"}"
: "${HOST:="windows"}"
: "${PROCESS:="qemu"}"

# --- 2. HARDWARE-RESOURCEN ---
: "${RAM_SIZE:="4G"}"
: "${CPU_CORES:="2"}"
: "${DISK_SIZE:="64G"}"
: "${ALLOCATE:="N"}"
: "${MACHINE:="q35"}"
: "${PLATFORM:="x64"}"
: "${SOCKETS:="1"}"
: "${CORES:="$CPU_CORES"}"
: "${THREADS:="1"}"
: "${ARCH:="x86_64"}"
export ARCH

# --- 3. BOOT & ISO OPTIONEN ---
: "${BOOT:=""}"
: "${BOOT_INDEX:="9"}"
: "${BOOT_MODE:="uefi"}"
: "${RECOVERY:="N"}"

# --- 4. NETZWERK (Wichtig für dnsmasq!) ---
: "${DHCP:="N"}"
: "${DNS:="8.8.8.8"}"
: "${BRIDGE:=""}"
: "${NET_DEVICE:="eth0"}"
: "${GATEWAY:="172.17.0.1"}"
: "${ADDRESS:="172.17.0.2"}"
: "${NET_MASK:="255.255.255.0"}"
: "${NET_SPEED:="1000"}"
: "${NET_MTU:="1500"}"
: "${NET_MODEL:="ve1000"}"

# --- 5. PORTS & DISCOVERY ---
: "${VNC_PORT:="5900"}"
: "${WSS_PORT:="8001"}"
: "${WEB_PORT:="8006"}"
: "${MON_PORT:="8000"}"
: "${WSD_PORT:="3702"}"
: "${LLMNR_PORT:="5355"}"
: "${NBNS_PORT:="137"}"

# --- 6. SPEICHERPFADE ---
: "${STORAGE:="/storage"}"
: "${DIST:="/run/shm"}"
: "${TMP:="$STORAGE/tmp"}"
DNSMASQ="/bin/true"

# --- Beschleuniger-Fix für Windows 24H2 ---
# : "${ACCEL:="whpx:tcg"}"
: "${KVM:="Y"}"
: "${CPU_MODEL:="qemu64"}" # Manche WHPX-Versionen mögen 'host' nicht, 'qemu64' ist sicherer

export ACCEL KVM CPU_MODEL

# --- Prozess- & QEMU-Parameter ---
# Wir prüfen erst, ob KVM im Linux-Kernel (WSL) verfügbar ist
if [ -e /dev/kvm ]; then
    info "KVM device found, enabling hardware acceleration."
    KVM="Y"
    ACCEL="kvm"
    CPU_MODEL="host"
else
    # Wenn KVM fehlt (typisch für Windows 24H2), versuchen wir WHPX oder TCG
    warn "KVM device not found! Using fallback acceleration."
    KVM="N"
    
    # WHPX funktioniert am besten, wenn wir QEMU sagen, er soll Windows-Beschleunigung nutzen
    # Wenn wir in Docker auf WSL2 sind, ist 'whpx' oft über den Host-Pass-Through erreichbar
    # Falls das fehlschlägt, nutzt QEMU automatisch TCG (Emulation)
    ACCEL="whpx:tcg,thread=multi"
    
    # 'host' funktioniert bei Emulation oft nicht, daher ein sicheres Modell
    CPU_MODEL="qemu64"
fi

# Jetzt exportieren wir die finalen Werte, damit boot.sh sie sieht
export KVM ACCEL CPU_MODEL

# Hilfsvariablen für QEMU (falls noch nicht gesetzt)
: "${ARGUMENTS:=""}"
: "${CPU_FLAGS:=""}"
: "${KVM_OPTS:=""}"
export ARGUMENTS CPU_FLAGS KVM_OPTS

echo "Basis-Variablen geladen. Beschleuniger: $ACCEL (KVM=$KVM)"

# --- 7. EXPORTIERE ALLES FÜR SUB-SKRIPTE ---
export APP VERSION ENGINE ROOTLESS PRIVILEGED HOST PROCESS RAM_SIZE CPU_CORES DISK_SIZE ALLOCATE KVM MACHINE PLATFORM BOOT BOOT_INDEX BOOT_MODE RECOVERY DHCP DNS BRIDGE NET_DEVICE GATEWAY ADDRESS NET_MASK NET_SPEED NET_MTU NET_MODEL VNC_PORT WSS_PORT WEB_PORT MON_PORT WSD_PORT LLMNR_PORT NBNS_PORT STORAGE DIST TMP

# Bereinigt die Variablen von unsichtbaren Windows-Steuerzeichen
BRIDGE=$(echo "${BRIDGE}" | tr -d '\r')
DNS=$(echo "${DNS}" | tr -d '\r')
ADDRESS=$(echo "${ADDRESS}" | tr -d '\r')
DHCP=$(echo "${DHCP}" | tr -d '\r')

# Die wichtigsten Pfade
export STORAGE="/storage"
export TMP="$STORAGE/tmp"

# Standard-Werte für die Installation
export VERSION="win11"
export LANGUAGE="en-US"
export DETECTED=""
export CUSTOM=""
export DEBUG="${DEBUG:-N}"
export INTERACTIVE="${INTERACTIVE:-N}"

echo "Basis-Variablen geladen."

: "${WIDTH:=""}"
: "${HEIGHT:=""}"
: "${VERIFY:=""}"
: "${REGION:=""}"
: "${MANUAL:=""}"
: "${REMOVE:=""}"
: "${VERSION:=""}"
: "${DETECTED:=""}"
: "${KEYBOARD:=""}"
: "${LANGUAGE:=""}"
: "${USERNAME:=""}"
: "${PASSWORD:=""}"

MIRRORS=4
PLATFORM="x64"

# Radikale Säuberung aller Netzwerk-Variablen von Windows-Resten
BRIDGE=$(echo "${BRIDGE:-br0}" | tr -d '\r')
DHCP=$(echo "${DHCP:-N}" | tr -d '\r')
DNS=$(echo "${DNS:-8.8.8.8}" | tr -d '\r')
GATEWAY=$(echo "${GATEWAY:-172.17.0.1}" | tr -d '\r')
ADDRESS=$(echo "${ADDRESS:-172.17.0.2}" | tr -d '\r')


parseVersion() {

  if [[ "${VERSION}" == \"*\" || "${VERSION}" == \'*\' ]]; then
    VERSION="${VERSION:1:-1}"
  fi

  [ -z "$VERSION" ] && VERSION="win11"

  case "${VERSION,,}" in
    "11" | "11p" | "win11" | "pro11" | "win11p" | "windows11" | "windows 11" )
      VERSION="win11x64"
      ;;
    "11e" | "win11e" | "windows11e" | "windows 11e" | "win11x64-enterprise-eval" )
      VERSION="win11x64-enterprise-eval"
      ;;
  esac

  return 0
}

setOwner() {
  local file="$1"
  # Wenn die Datei nicht existiert, machen wir nichts
  [ ! -e "$file" ] && return 0
  
  # Setzt den Besitzer auf Root (Standard im Container)
  # Falls Fehler auftreten, gibt die Funktion 1 zurück
  chown root:root "$file" || return 1
  chmod 660 "$file" || return 1
  
  return 0
}

getLanguage() {

  local id="$1"
  local ret="$2"
  local lang=""
  local desc=""
  local culture=""

  case "${id,,}" in
    "ar" | "ar-"* )
      lang="Arabic"
      desc="$lang"
      culture="ar-SA" ;;
    "bg" | "bg-"* )
      lang="Bulgarian"
      desc="$lang"
      culture="bg-BG" ;;
    "cs" | "cs-"* | "cz" | "cz-"* )
      lang="Czech"
      desc="$lang"
      culture="cs-CZ" ;;
    "da" | "da-"* | "dk" | "dk-"* )
      lang="Danish"
      desc="$lang"
      culture="da-DK" ;;
    "de" | "de-"* )
      lang="German"
      desc="$lang"
      culture="de-DE" ;;
    "el" | "el-"* | "gr" | "gr-"* )
      lang="Greek"
      desc="$lang"
      culture="el-GR" ;;
    "gb" | "en-gb" )
      lang="English International"
      desc="English"
      culture="en-GB" ;;
    "en" | "en-"* )
      lang="English"
      desc="English"
      culture="en-US" ;;
    "mx" | "es-mx" )
      lang="Spanish (Mexico)"
      desc="Spanish"
      culture="es-MX" ;;
    "es" | "es-"* )
      lang="Spanish"
      desc="$lang"
      culture="es-ES" ;;
    "et" | "et-"* )
      lang="Estonian"
      desc="$lang"
      culture="et-EE" ;;
    "fi" | "fi-"* )
      lang="Finnish"
      desc="$lang"
      culture="fi-FI" ;;
    "ca" | "fr-ca" )
      lang="French Canadian"
      desc="French"
      culture="fr-CA" ;;
    "fr" | "fr-"* )
      lang="French"
      desc="$lang"
      culture="fr-FR" ;;
    "he" | "he-"* | "il" | "il-"* )
      lang="Hebrew"
      desc="$lang"
      culture="he-IL" ;;
    "hr" | "hr-"* | "cr" | "cr-"* )
      lang="Croatian"
      desc="$lang"
      culture="hr-HR" ;;
    "hu" | "hu-"* )
      lang="Hungarian"
      desc="$lang"
      culture="hu-HU" ;;
    "it" | "it-"* )
      lang="Italian"
      desc="$lang"
      culture="it-IT" ;;
    "ja" | "ja-"* | "jp" | "jp-"* )
      lang="Japanese"
      desc="$lang"
      culture="ja-JP" ;;
    "ko" | "ko-"* | "kr" | "kr-"* )
      lang="Korean"
      desc="$lang"
      culture="ko-KR" ;;
    "lt" | "lt-"* )
      lang="Lithuanian"
      desc="$lang"
      culture="lv-LV" ;;
    "lv" | "lv-"* )
      lang="Latvian"
      desc="$lang"
      culture="lt-LT" ;;
    "nb" | "nb-"* |"nn" | "nn-"* | "no" | "no-"* )
      lang="Norwegian"
      desc="$lang"
      culture="nb-NO" ;;
    "nl" | "nl-"* )
      lang="Dutch"
      desc="$lang"
      culture="nl-NL" ;;
    "pl" | "pl-"* )
      lang="Polish"
      desc="$lang"
      culture="pl-PL" ;;
    "br" | "pt-br" )
      lang="Brazilian Portuguese"
      desc="Portuguese"
      culture="pt-BR" ;;
    "pt" | "pt-"* )
      lang="Portuguese"
      desc="$lang"
      culture="pt-BR" ;;
    "ro" | "ro-"* )
      lang="Romanian"
      desc="$lang"
      culture="ro-RO" ;;
    "ru" | "ru-"* )
      lang="Russian"
      desc="$lang"
      culture="ru-RU" ;;
    "sk" | "sk-"* )
      lang="Slovak"
      desc="$lang"
      culture="sk-SK" ;;
    "sl" | "sl-"* | "si" | "si-"* )
      lang="Slovenian"
      desc="$lang"
      culture="sl-SI" ;;
    "sr" | "sr-"* )
      lang="Serbian Latin"
      desc="Serbian"
      culture="sr-Latn-RS" ;;
    "sv" | "sv-"* | "se" | "se-"* )
      lang="Swedish"
      desc="$lang"
      culture="sv-SE" ;;
    "th" | "th-"* )
      lang="Thai"
      desc="$lang"
      culture="th-TH" ;;
    "tr" | "tr-"* )
      lang="Turkish"
      desc="$lang"
      culture="tr-TR" ;;
    "ua" | "ua-"* | "uk" | "uk-"* )
      lang="Ukrainian"
      desc="$lang"
      culture="uk-UA" ;;
    "hk" | "zh-hk" | "cn-hk" )
      lang="Chinese (Traditional)"
      desc="Chinese HK"
      culture="zh-TW" ;;
    "tw" | "zh-tw" | "cn-tw" )
      lang="Chinese (Traditional)"
      desc="Chinese TW"
      culture="zh-TW" ;;
    "zh" | "zh-"* | "cn" | "cn-"* )
      lang="Chinese (Simplified)"
      desc="Chinese"
      culture="zh-CN" ;;
  esac

  case "${ret,,}" in
    "desc" ) echo "$desc" ;;
    "name" ) echo "$lang" ;;
    "culture" ) echo "$culture" ;;
    *) echo "$desc";;
  esac

  return 0
}

parseLanguage() {

  REGION="${REGION//_/-/}"
  KEYBOARD="${KEYBOARD//_/-/}"
  LANGUAGE="${LANGUAGE//_/-/}"

  [ -z "$LANGUAGE" ] && LANGUAGE="en"

  case "${LANGUAGE,,}" in
    "arabic" | "arab" ) LANGUAGE="ar" ;;
    "bulgarian" | "bu" ) LANGUAGE="bg" ;;
    "chinese" | "cn" ) LANGUAGE="zh" ;;
    "croatian" | "cr" | "hrvatski" ) LANGUAGE="hr" ;;
    "czech" | "cz" | "cesky" ) LANGUAGE="cs" ;;
    "danish" | "dk" | "danske" ) LANGUAGE="da" ;;
    "dutch" | "nederlands" ) LANGUAGE="nl" ;;
    "english" | "gb" | "british" ) LANGUAGE="en" ;;
    "estonian" | "eesti" ) LANGUAGE="et" ;;
    "finnish" | "suomi" ) LANGUAGE="fi" ;;
    "french" | "français" | "francais" ) LANGUAGE="fr" ;;
    "german" | "deutsch" ) LANGUAGE="de" ;;
    "greek" | "gr" ) LANGUAGE="el" ;;
    "hebrew" | "il" ) LANGUAGE="he" ;;
    "hungarian" | "magyar" ) LANGUAGE="hu" ;;
    "italian" | "italiano" ) LANGUAGE="it" ;;
    "japanese" | "jp" ) LANGUAGE="ja" ;;
    "korean" | "kr" ) LANGUAGE="ko" ;;
    "latvian" | "latvijas" ) LANGUAGE="lv" ;;
    "lithuanian" | "lietuvos" ) LANGUAGE="lt" ;;
    "norwegian" | "no" | "nb" | "norsk" ) LANGUAGE="nn" ;;
    "polish" | "polski" ) LANGUAGE="pl" ;;
    "portuguese" | "pt" | "br" ) LANGUAGE="pt-br" ;;
    "português" | "portugues" ) LANGUAGE="pt-br" ;;
    "romanian" | "română" | "romana" ) LANGUAGE="ro" ;;
    "russian" | "ruski" ) LANGUAGE="ru" ;;
    "serbian" | "serbian latin" ) LANGUAGE="sr" ;;
    "slovak" | "slovenský" | "slovensky" ) LANGUAGE="sk" ;;
    "slovenian" | "si" | "slovenski" ) LANGUAGE="sl" ;;
    "spanish" | "espanol" | "español" ) LANGUAGE="es" ;;
    "swedish" | "se" | "svenska" ) LANGUAGE="sv" ;;
    "turkish" | "türk" | "turk" ) LANGUAGE="tr" ;;
    "thai" ) LANGUAGE="th" ;;
    "ukrainian" | "ua" ) LANGUAGE="uk" ;;
  esac

  local culture
  culture=$(getLanguage "$LANGUAGE" "culture")
  [ -n "$culture" ] && return 0

  error "Invalid LANGUAGE specified, value \"$LANGUAGE\" is not recognized!"
  return 1
}

printVersion() {

  local id="$1"
  local desc="$2"

  case "${id,,}" in
    "win11"* ) desc="Windows 11" ;;
  esac

  if [ -z "$desc" ]; then
    desc="Windows"
    [[ "${PLATFORM,,}" != "x64" ]] && desc+=" for ${PLATFORM}"
  fi

  echo "$desc"
  return 0
}

printEdition() {

  local id="$1"
  local desc="$2"
  local result=""
  local edition=""

  result=$(printVersion "$id" "x")
  [[ "$result" == "x" ]] && echo "$desc" && return 0

  case "${id,,}" in
    *"-enterprise" )
      edition="Enterprise"
      ;;
    *"-enterprise-eval" )
      edition="Enterprise (Evaluation)"
      ;;
  esac

  [ -n "$edition" ] && result+=" $edition"

  echo "$result"
  return 0
}

fromName() {

  local id=""
  local name="$1"
  local arch="$2"

  local add=""
  [[ "$arch" != "x64" ]] && add="$arch"

  case "${name,,}" in
    *"windows 11"* ) id="win11${arch}" ;;
  esac

  echo "$id"
  return 0
}

getVersion() {

  local id
  local name="$1"
  local arch="$2"

  id=$(fromName "$name" "$arch")

  case "${id,,}" in
    "win11"* )
       case "${name,,}" in
          *" enterprise evaluation"* ) id="$id-enterprise-eval" ;;
          *" enterprise"* ) id="$id-enterprise" ;;
        esac
      ;;
  esac

  echo "$id"
  return 0
}

addFolder() {

  local src="$1"
  local folder="/oem"

  [ ! -d "$folder" ] && folder="/OEM"
  [ ! -d "$folder" ] && folder="$STORAGE/oem"
  [ ! -d "$folder" ] && folder="$STORAGE/OEM"
  [ ! -d "$folder" ] && return 0

  local msg="Adding OEM folder to image..."
  info "$msg" && html "$msg"

  local dest="$src/\$OEM\$/\$1/OEM"
  mkdir -p "$dest" || return 1
  cp -Lr "$folder/." "$dest" || return 1

  local file
  file=$(find "$dest" -maxdepth 1 -type f -iname install.bat | head -n 1)
  [ -f "$file" ] && unix2dos -q "$file"

  return 0
}

# migrateFiles() {

#   local base="$1"
#   local version="$2"
#   local file=""

#   [ -f "$base" ] && return 0

#   [[ "${version,,}" == "tiny10" ]] && file="tiny10_x64_23h2.iso"
#   [[ "${version,,}" == "tiny11" ]] && file="tiny11_2311_x64.iso"
#   [[ "${version,,}" == "core11" ]] && file="tiny11_core_x64_beta_1.iso"
#   [[ "${version,,}" == "winxpx86" ]] && file="en_windows_xp_professional_with_service_pack_3_x86_cd_x14-80428.iso"
#   [[ "${version,,}" == "winvistax64" ]] && file="en_windows_vista_sp2_x64_dvd_342267.iso"
#   [[ "${version,,}" == "win7x64" ]] && file="en_windows_7_enterprise_with_sp1_x64_dvd_u_677651.iso"

#   [ ! -f "$STORAGE/$file" ] && return 0
#   mv -f "$STORAGE/$file" "$base" || return 1

#   return 0
# }

migrateFiles() {

  local base="$1"
  local version="$2"
  local file=""

  [ -f "$base" ] && return 0

  [ ! -f "$STORAGE/$file" ] && return 0
  mv -f "$STORAGE/$file" "$base" || return 1

  return 0
}

return 0
