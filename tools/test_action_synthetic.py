"""动作判据单元测试（合成样本，不需要真人）。

直接调用 `liveness.analyze_sequence()`，而不是复刻它的内部逻辑 ——
这样测的是**真正跑在服务端的那段代码**，包括基线取法、ROI 刚性跟踪、连续段判据。

覆盖两类错误：
    误放（该拒没拒）：静止照片、抖动照片、**做的次数不够**（要求两次只做一次）
    误拒（该过没过）：正常张嘴、张两次、**动作占满整段**

关于「动作占满整段」
--------------------
早期版本用序列自身的分位数当基线。学生若看到指令立刻张嘴并一直保持到采集结束，
「闭嘴基线」取到的其实是张嘴值，算出来的增幅≈0 → 判为没做动作。
真人明明做了，系统说他没做。现在前端会提交「指令出现前」的基线帧，这个用例专门守它。

局限：合成形变不等于真实肌肉运动，本测试只证明判据与阈值量级合理。
真人通过率必须在真实摄像头上验收（诊断小字 / GET /api/attendance/diagnostics）。
"""
import sys

from tools.synth import Face, sequence

from backend import liveness  # noqa: E402

N = 22


def main():
    face = Face()
    print(f"人脸框 {[round(v) for v in face.bbox]} 宽 {face.bbox[2] - face.bbox[0]:.0f}px  "
          f"序列 {N} 帧 ≈ {N * 0.1:.1f} 秒")

    total = passed = 0

    def check(name, cond, extra=""):
        nonlocal total, passed
        total += 1
        passed += 1 if cond else 0
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")

    def run(frames, action, ref=None):
        return liveness.analyze_sequence(frames, action, ref=ref if ref is not None else face.img)

    def brief(r):
        d = r["metrics"].get("action_detail", {})
        keep = {k: v for k, v in d.items() if k != "hint"}
        return f"reason={r['reason']} {keep}"

    # ---- 1. 静止画面：两条指令都必须拒 ----
    still = sequence(face, None, n=N)
    print("\n[1] 静止画面（相当于举着一张照片不动）")
    r1 = run(still, "mouth")
    check("静止画面不算张嘴", not r1["ok"], brief(r1))
    r2 = run(still, "mouth2")
    check("静止画面不算张两次", not r2["ok"], brief(r2))
    check("拒绝原因都是动作缺失",
          r1["reason"] == "action_missing" and r2["reason"] == "action_missing")

    # ---- 2. 做对了：必须通过 ----
    print("\n[2] 合成张嘴一次（保持 8 帧 ≈ 0.8 秒）")
    one = sequence(face, "mouth", n=N, start=5, length=8)
    r3 = run(one, "mouth")
    check("张嘴一次被判为通过", r3["ok"], brief(r3))

    print("\n[3] 合成张嘴两次（第 3~6 帧、第 9~13 帧各张一次）")
    two = sequence(face, "mouth2", n=N)
    r4 = run(two, "mouth2")
    check("张两次被判为通过", r4["ok"], brief(r4))
    r4b = run(two, "mouth")   # 做多了不算错，人不能因为多做一次被判失败
    check("张两次也满足「张一次」的指令", r4b["ok"], brief(r4b))

    # ---- 4. 次数不够：这是随机性的全部价值所在 ----
    print("\n[4] 次数不够：做了张嘴，但服务端要求张两次")
    r5 = run(one, "mouth2")
    check("只张一次满足不了「张两次」", not r5["ok"], brief(r5))
    check("提示里说明了次数不足",
          "次" in (r5["metrics"].get("action_detail", {}).get("hint") or ""),
          r5["metrics"].get("action_detail", {}).get("hint"))

    # ---- 5. 动作占满整段：早期版本在这里必挂 ----
    print("\n[5] 动作占满整段（学生看到指令就张嘴并一直保持到结束）")
    full = sequence(face, "mouth", n=N, start=0, length=N)
    r6 = run(full, "mouth")
    check("占满整段也能判为张嘴", r6["ok"], brief(r6))
    r6b = liveness.analyze_sequence(full, "mouth", ref=None)   # 不给基线帧的兜底路径
    print(f"    兜底路径（无 ref，按序列分两堆）ok={r6b['ok']}  ← 这里本来就判不出来，仅记录")

    # ---- 6. 抖动照片：刚性跟踪必须把抖动磨平 ----
    print("\n[6] 抖动照片（整帧平移 ±6px，模拟手持）")
    shaky = face.shaken(n=N)
    r7 = run(shaky, "mouth", ref=shaky[0])
    check("抖动照片不算张嘴", not r7["ok"], brief(r7))
    print(f"    嘴暗度序列={r7['metrics']['mouth_dark']}")
    print(f"    ROI 基准  ={r7['metrics']['ref_base']}")

    print(f"\n结果: {passed}/{total} 通过")
    print("\n提醒：合成样本只证明判据有效。真实摄像头下的通过率需要现场验收。")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
