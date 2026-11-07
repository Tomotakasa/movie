#!/bin/bash
# 手ブレ補正アプリの起動スクリプト(Finder でダブルクリックで起動できます)
cd "$(dirname "$0")" || exit 1
export PYTHONIOENCODING=utf-8
export LANG="${LANG:-ja_JP.UTF-8}"

fail() {
  echo ""
  echo "❌ $1"
  echo "この画面のスクリーンショットを送ってください。Enter キーで閉じます。"
  read -r _
  exit 1
}

if ! command -v python3 >/dev/null 2>&1; then
  osascript -e 'display dialog "Python 3 が見つかりません。\nターミナルで「xcode-select --install」を実行するか、python.org から Python をインストールしてください。" buttons {"OK"} with icon caution' >/dev/null
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "初回セットアップ中です(数分かかります)…"
  python3 -m venv .venv || fail "仮想環境の作成に失敗しました"
fi

# requirements.txt が変わったときだけ再インストール
if ! cmp -s requirements.txt .venv/.installed 2>/dev/null; then
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  # --no-compile: PySide6 同梱のテンプレートファイルで事前コンパイルが失敗するのを避ける
  if ! .venv/bin/python -m pip install --no-compile -r requirements.txt; then
    rm -rf .venv   # 中途半端な状態を残さない(次回はやり直し)
    fail "ライブラリのインストールに失敗しました"
  fi
  cp requirements.txt .venv/.installed
fi

.venv/bin/python -m stabilizer "$@" || fail "アプリがエラーで終了しました"
