#!/usr/bin/env bash
# Download the pinned public SegFormer optic-disc/cup model (Apache-2.0) and verify it.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 WEIGHTS_DIR" >&2
  exit 2
fi

destination="$(realpath -m "$1")/external/optic_disc"
if [[ -e "${destination}" ]]; then
  echo "Refusing to overwrite ${destination}" >&2
  exit 2
fi
mkdir -p "${destination}"

base="https://huggingface.co/pamixsun/segformer_for_optic_disc_cup_segmentation/resolve/e1698e9f52e24cb6a7b2fecab4688852b89f77ef"
for name in config.json preprocessor_config.json pytorch_model.bin; do
  curl -fL "${base}/${name}" -o "${destination}/${name}"
done

(cd "${destination}" && sha256sum -c - <<'SUMS'
e3639987b9b21f8726d8b8abad849b2366936ff5ed042c255ef14390243ef858  config.json
b265f6d53be9b46b2d9f9a16c70cf7648196d57f31192f5bb67212031fea12ef  preprocessor_config.json
1a1a4154b5a6d6abf4333a12c3903cc84a16d57da7c6d9eaffec3748c3e552e3  pytorch_model.bin
SUMS
)
echo "Prepared pinned SegFormer files in ${destination}"
