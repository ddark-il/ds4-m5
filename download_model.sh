#!/bin/sh
set -e

REPO="ddark-il/deepseek-v4-gguf"
MODEL_FILE="DeepSeek-V4-Flash-Layers1-2.37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
DSPARK_SUPPORT_FILE="DeepSeek-V4-Flash-DSpark-support.gguf"

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT_DIR=${DS4_GGUF_DIR:-"$ROOT/gguf"}
case "$OUT_DIR" in
    /*) ;;
    *) OUT_DIR="$ROOT/$OUT_DIR" ;;
esac
TOKEN=${HF_TOKEN:-}

usage() {
    cat <<EOF
DwarfStar GGUF downloader (DeepSeek V4 Flash, Apple M5 Max)

Downloads from https://huggingface.co/$REPO

Usage:
  ./download_model.sh model [--token TOKEN]
  ./download_model.sh dspark-support [--token TOKEN]

Targets:

  model
       DeepSeek V4 Flash mixed-quant imatrix GGUF (deg8): Q4_K routed experts
       on layers 1-2 + 37-42, IQ2_XXS gate/up + Q2_K down on the other layers,
       Q8_0 attention / shared expert / output. About 94 GB on disk.
       Recommended default for a 128 GB Apple M5 Max. Links ./ds4flash.gguf to it.

  dspark-support
       Optional DSpark speculative-decode support GGUF, about 6 GB. Enable it
       with --dspark and --mtp when running ds4 / ds4-agent.

Options:
  --token TOKEN  Hugging Face token. Otherwise HF_TOKEN or the local Hugging
                 Face token cache is used if present.

Environment:
  DS4_GGUF_DIR   Directory used for downloaded GGUF files. Default: ./gguf

After the model download the script links:
  ./ds4flash.gguf -> <download directory>/<model>

Then the default commands work:
  ./ds4 -p "Hello"
  ./ds4-agent

After downloading DSpark support, enable it explicitly in greedy mode:
  ./ds4 --dspark -m ./ds4flash.gguf --mtp <download directory>/$DSPARK_SUPPORT_FILE --temp 0
EOF
}

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

MODEL=$1
shift
LINK_MODEL=1

case "$MODEL" in
    model) ;;
    dspark-support) MODEL_FILE=$DSPARK_SUPPORT_FILE; LINK_MODEL=0 ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown target: $MODEL" >&2
        echo >&2
        usage >&2
        exit 1
        ;;
esac

while [ $# -gt 0 ]; do
    case "$1" in
        --token)
            shift
            if [ $# -eq 0 ]; then
                echo "Missing value after --token" >&2
                exit 1
            fi
            TOKEN=$1
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
    shift
done

if [ -z "$TOKEN" ] && [ -s "$HOME/.cache/huggingface/token" ]; then
    TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi

download_one() {
    file=$1
    out="$OUT_DIR/$file"
    part="$out.part"
    url="https://huggingface.co/$REPO/resolve/main/$file"

    mkdir -p "$(dirname "$out")"

    if [ -s "$out" ]; then
        echo "Already downloaded: $out"
        return
    fi

    echo "Downloading $file"
    echo "from https://huggingface.co/$REPO"
    echo "If the download stops, run the same command again to resume it."

    if [ -n "$TOKEN" ]; then
        curl -fL --progress-meter -C - -H "Authorization: Bearer $TOKEN" -o "$part" "$url"
    else
        curl -fL --progress-meter -C - -o "$part" "$url"
    fi

    mv "$part" "$out"
}

download_one "$MODEL_FILE"

if [ "$MODEL" = "dspark-support" ]; then
    echo
    echo "DSpark support downloaded. Enable it explicitly in greedy mode:"
    echo "  ./ds4 --dspark -m ./ds4flash.gguf --mtp $OUT_DIR/$DSPARK_SUPPORT_FILE --temp 0"
elif [ "$LINK_MODEL" -eq 1 ]; then
    cd "$ROOT"
    ln -sfn "$OUT_DIR/$MODEL_FILE" ds4flash.gguf
    echo "Linked ./ds4flash.gguf -> $OUT_DIR/$MODEL_FILE"
fi

echo
echo "Done."
