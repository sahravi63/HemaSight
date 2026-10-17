#!/usr/bin/env bash
# One-time setup for an IBM LinuxONE Community Cloud server (s390x).
# Usage:  bash setup_server.sh
set -euo pipefail

echo "== HemaSight setup on $(uname -m) =="

if command -v apt-get >/dev/null; then
  sudo apt-get update -y
  # distro packages are pre-built for s390x, which avoids slow source builds
  sudo apt-get install -y python3 python3-venv python3-pip python3-numpy python3-pandas \
    python3-sklearn python3-pil python3-flask python3-joblib python3-openpyxl unzip \
    iptables-persistent || sudo apt-get install -y python3 python3-venv python3-pip \
    python3-numpy python3-pandas python3-sklearn python3-pil python3-flask python3-joblib \
    python3-openpyxl unzip
elif command -v dnf >/dev/null; then
  sudo dnf install -y python3 python3-pip python3-numpy python3-pandas python3-scikit-learn \
    python3-pillow python3-flask python3-joblib python3-openpyxl unzip || true
fi

# venv that can still see the distro's pre-built packages
python3 -m venv --system-site-packages ~/hemasight-venv
source ~/hemasight-venv/bin/activate
pip install --upgrade pip >/dev/null

# anything the distro did not provide
python3 - <<'EOF' || pip install numpy pandas scikit-learn pillow flask joblib openpyxl
import numpy, pandas, sklearn, PIL, flask, joblib, openpyxl
EOF

# optional: IBM Snap ML (fast tree models, has s390x builds)
pip install snapml >/dev/null 2>&1 && echo "snapml: installed" || echo "snapml: not available, continuing without it"

python3 - <<'EOF'
import numpy, pandas, sklearn, PIL, flask
print("numpy", numpy.__version__, "| pandas", pandas.__version__,
      "| scikit-learn", sklearn.__version__, "| pillow", PIL.__version__, "| flask", flask.__version__)
EOF

# open port 5000 for the demo app (only SSH is open by default)
if ! sudo iptables -C INPUT -p tcp --dport 5000 -j ACCEPT 2>/dev/null; then
  sudo iptables -I INPUT -p tcp --dport 5000 -j ACCEPT
fi
if [ -d /etc/iptables ]; then sudo sh -c 'iptables-save > /etc/iptables/rules.v4'; fi

echo
echo "Done. Next:"
echo "  source ~/hemasight-venv/bin/activate"
echo "  python3 train.py --data ~/data --out models"
echo "  python3 app.py --models models --port 5000   # then open http://<server-ip>:5000"
