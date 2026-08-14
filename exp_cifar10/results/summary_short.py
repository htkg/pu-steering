import json
import os


def main() -> None:
    cwd = os.getcwd()
    for name in sorted(os.listdir(cwd)):
        src = os.path.join(cwd, name, "summary.json")
        dst = os.path.join(cwd, name, "summary_short.json")

        if not os.path.isfile(src) or os.path.isfile(dst):
            continue
        with open(src) as f:
            s = json.load(f)

        s.pop("per_seed", None)
        if not s.get("config"):
            s.pop("config", None)

        with open(dst, "w") as f:
            json.dump(s, f, indent=2)
        print(f"  {name}/summary_short.json")


if __name__ == "__main__":
    main()
