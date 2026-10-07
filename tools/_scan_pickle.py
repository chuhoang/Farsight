"""Static scan of torch .pt pickles: list imported globals without unpickling (no code is executed)."""
import pickletools, sys, zipfile
for path in sys.argv[1:]:
    mods = set()
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if n.endswith(".pkl"):
                ops = list(pickletools.genops(z.read(n)))
                strs = []
                for op, arg, _ in ops:
                    if op.name == "GLOBAL":
                        mods.add(arg.replace(" ", "."))
                    elif op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
                        strs.append(arg)
                    elif op.name == "STACK_GLOBAL" and len(strs) >= 2:
                        mods.add(f"{strs[-2]}.{strs[-1]}")
    bad = sorted(m for m in mods if m.split(".")[0] in {"os", "subprocess", "builtins", "posix", "nt", "sys", "socket", "shutil", "runpy", "importlib", "pty", "webbrowser"} or m.endswith((".eval", ".exec", ".system", ".__import__")))
    print("  builtins/codecs used:", sorted(m for m in mods if m.startswith(("__builtin__", "_codecs", "builtins"))))
    tops = sorted({m.split(".")[0] for m in mods})
    print(path.split("/")[-1], "| globals:", len(mods), "| top-level modules:", tops, "| SUSPICIOUS:", bad or "none")
