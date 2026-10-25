import json, glob, os

for f in sorted(glob.glob("results/*.json")):
    r = json.load(open(f))
    h = r["history"]
    last = h[-1]
    mem = r.get("peak_gpu_mem_mb", 0)
    ep_t = sum(x.get("epoch_time_s", 0) for x in h) / len(h)
    print(f"{r['name']:<22} test={r['test_acc']:.4f} | "
          f"last epoch: tr_loss={last['train_loss']:.4f} tr_acc={last['train_acc']:.4f} "
          f"val_loss={last['val_loss']:.4f} val_acc={last['val_acc']:.4f} | "
          f"train={r['train_time_s']:.1f}s ({ep_t:.1f}s/ep) mem={mem:.0f}MB")
