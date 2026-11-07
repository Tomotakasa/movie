"""起動: python -m stabilizer [動画]   /  CLI: python -m stabilizer --cli 入力 出力 [--strength 秒] [--tripod]"""
import argparse
import sys


def cli(argv):
    from . import core
    ap = argparse.ArgumentParser(prog="stabilizer --cli", description="手ブレ補正(コマンドライン)")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--strength", type=float, default=0.75, help="補正の強さ(秒) 既定 0.75")
    ap.add_argument("--tripod", action="store_true", help="三脚固定モード")
    ap.add_argument("--no-rotation", action="store_true", help="回転を補正しない")
    ap.add_argument("--codec", default="h264", choices=list(core.CODECS))
    a = ap.parse_args(argv)

    def bar(label):
        return lambda i, n: print(f"\r{label} {i}/{n}", end="", flush=True)
    an = core.analyze(a.input, progress=bar("解析"), keep_previews=False)
    print()
    st = core.Stabilization(an, core.Settings(mode="tripod" if a.tripod else "smooth",
                                              smoothness_sec=a.strength, correct_rotation=not a.no_rotation))
    b, af, r = st.shake_metrics()
    print(f"揺れ {b:.2f}px → {af:.2f}px(低減 {r * 100:.1f}%) 拡大率 {st.zoom * 100:.1f}%")
    core.export(st, a.output, a.codec, progress=bar("書き出し"))
    print("\n完了:", a.output)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        cli(sys.argv[2:])
    else:
        from .gui import main
        sys.exit(main())
