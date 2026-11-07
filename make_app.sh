#!/bin/bash
# 「手ブレ補正.app」を ~/Applications に作成します(Launchpad や Dock から起動できるように)
# 使い方: ターミナルで  ./make_app.sh
set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
APP="$HOME/Applications/手ブレ補正.app"
mkdir -p "$APP/Contents/MacOS"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>手ブレ補正</string>
  <key>CFBundleIdentifier</key><string>local.stabilizer</string>
  <key>CFBundleExecutable</key><string>launcher</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
cat > "$APP/Contents/MacOS/launcher" <<LAUNCH
#!/bin/bash
exec "$REPO/run.command" "\$@" >/tmp/stabilizer.log 2>&1
LAUNCH
chmod +x "$APP/Contents/MacOS/launcher"
echo "作成しました: $APP"
echo "※ 初回は run.command を一度ダブルクリックしてセットアップを済ませてから使ってください。"
