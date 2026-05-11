#!/usr/bin/env bash
set -euo pipefail

SOUNDFONT_DIR="soundfont"
SOUNDFONT_NAME="st_concert.sf2"

run_cmd() {
    echo "  $*"
    "$@"
}

run_all() {
    local -a cmds=("$@")
    for cmd in "${cmds[@]}"; do
        echo "  $cmd"
    done
    echo ""
    for cmd in "${cmds[@]}"; do
        eval "$cmd"
    done
}

detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif command -v apt-get &>/dev/null; then
        echo "debian"
    else
        echo "unknown"
    fi
}

install_debian() {
    echo "Installing system dependencies (Debian/Ubuntu):"
    run_all \
        "sudo apt-get update -qq" \
        "sudo apt-get install -y lilypond fluidsynth ffmpeg fluid-soundfont-gm"
}

install_macos() {
    echo "Installing system dependencies (macOS):"
    if ! command -v brew &>/dev/null; then
        echo "Homebrew is not installed. Install it from https://brew.sh and re-run this script."
        exit 1
    fi
    run_all "brew install lilypond fluid-synth ffmpeg"
}

setup_soundfont_debian() {
    local system_sf2="/usr/share/sounds/sf2/FluidR3_GM.sf2"
    if [[ ! -f "$system_sf2" ]]; then
        echo "Warning: expected soundfont not found at $system_sf2"
        return
    fi
    mkdir -p "$SOUNDFONT_DIR"
    if [[ ! -e "$SOUNDFONT_DIR/$SOUNDFONT_NAME" ]]; then
        ln -s "$system_sf2" "$SOUNDFONT_DIR/$SOUNDFONT_NAME"
        echo "Soundfont symlinked: $SOUNDFONT_DIR/$SOUNDFONT_NAME -> $system_sf2"
    else
        echo "Soundfont already present at $SOUNDFONT_DIR/$SOUNDFONT_NAME, skipping."
    fi
}

setup_soundfont_macos() {
    echo ""
    echo "macOS: no system soundfont is installed automatically."
    echo "Download GeneralUser GS from https://schristiancollins.com/generaluser.php"
    echo "and place (or symlink) it at: $SOUNDFONT_DIR/$SOUNDFONT_NAME"
    mkdir -p "$SOUNDFONT_DIR"
}

OS=$(detect_os)

case "$OS" in
    debian)
        install_debian
        setup_soundfont_debian
        ;;
    macos)
        install_macos
        setup_soundfont_macos
        ;;
    *)
        echo "Unsupported OS. Install lilypond, fluidsynth, and ffmpeg manually."
        echo "Then place a .sf2 soundfont at $SOUNDFONT_DIR/$SOUNDFONT_NAME"
        exit 1
        ;;
esac

echo ""
echo "Done. External dependencies are installed."
