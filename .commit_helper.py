import subprocess

def run(args, **kw):
    return subprocess.run(args, capture_output=True, text=True, check=True, **kw).stdout.strip()

tree = run(["git", "write-tree"])
parent = run(["git", "rev-parse", "HEAD"])
msg = "humanize comments and prune unused files\n"
new = subprocess.run(
    ["git", "commit-tree", tree, "-p", parent],
    input=msg,
    capture_output=True,
    text=True,
    check=True,
).stdout.strip()
run(["git", "update-ref", "HEAD", new])
print("new commit:", new)
print(run(["git", "log", "-1", "--format=fuller"]))
