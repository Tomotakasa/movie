#!/bin/bash
# 手ブレ補正アプリの起動スクリプト(Finder でダブルクリックで起動できます)
cd "$(dirname "$0")" || exit 1

if ! command -v python3 >/dev/null 2>&1; then
  osascript -e 'display dialog "Python 3 が見つかりません。\nターミナルで「xcode-select --install」を実行するか、python.org から Python をインストールしてください。" buttons {"OK"} with icon caution' >/dev/null
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "初回セットアップ中です(数分かかります)…"
  python3 -m venv .venv || exit 1
fi

# requirements.txt が変わったときだけ再インストール
if ! cmp -s requirements.txt .venv/.installed 2>/dev/null; then
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  .venv/bin/python -m pip install -r requirements.txt || exit 1
  cp requirements.txt .venv/.installed
fi

exec .venv/bin/python -m stabilizer "$@"
