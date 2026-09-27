"""Build and run an isolated, loopback-only Arena shard for development.

Never reads/copies another shard's Saves. Test account passwords are generated
into a mode-0600 file in the separate runtime directory. UO assets stay external.
"""
import argparse
import json
import os
import secrets
import shutil
import subprocess
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--servuo", type=Path, default=Path("../servuo"))
ap.add_argument("--data", type=Path, required=True)
ap.add_argument("--runtime", type=Path, default=Path("../runtime-local"))
ap.add_argument("--port", type=int, default=2597)
ap.add_argument("--no-build", action="store_true")
args = ap.parse_args()
source, runtime, data = args.servuo.resolve(), args.runtime.resolve(), args.data.resolve()
if runtime == source or (runtime / ".git").exists():
    ap.error("runtime must be a separate non-repository directory")
if not data.is_dir() or not (source / "ServUO.sln").is_file():
    ap.error("ServUO source and UO data directories must exist")
if runtime.exists() and any(runtime.iterdir()) and not (runtime / ".anima-arena-local").exists():
    ap.error("refusing to overwrite an existing directory not created by this launcher")
if not args.no_build:
    subprocess.run(["dotnet", "build", "-c", "Release"], cwd=source, check=True)
runtime.mkdir(parents=True, exist_ok=True)
(runtime / ".anima-arena-local").touch()
for name in ("Config", "Data"):
    if not (runtime / name).exists():
        shutil.copytree(source / name, runtime / name)
for name in ("ServUO.exe", "Scripts.dll", "Ultima.dll", "ServUO.exe.config"):
    shutil.copy2(source / name, runtime / name)
config = runtime / "Config"
(config / "Compiler.cfg").write_text("Dynamic=false\n")
(config / "DataPath.cfg").write_text(f"CustomPath={data}\n")
(config / "Server.cfg").write_text(f"Name=UO Tavern Arena Local\nListen=127.0.0.1\nAddress=127.0.0.1\nPort={args.port}\nMaxAddressesPerIP=30\n")
(config / "Accounts.cfg").write_text("AccountsPerIp=30\nAutoCreateAccounts=True\nProtectPasswords=NewSecureCrypt\n")
(config / "AutoSave.cfg").write_text("Enabled=True\nFrequency=00:00:01:00\nWarningTime=00:00:00:00\nArchivesEnabled=False\n")
(config / "Arena.cfg").write_text("Enabled=true\nPeerAgents=false\nWelcomeOnLogin=true\nDomain=arena.uotavern.com\nBotAccounts=arena_bot_mage,arena_bot_warrior,arena_train_a,arena_train_b\nSelfPlay=true\n")
credentials = runtime / "test-accounts.json"
if not credentials.exists():
    accounts = {name: secrets.token_hex(14) for name in ("arena_admin", "arena_playtest", "arena_bot_mage", "arena_bot_warrior", "arena_train_a", "arena_train_b")}
    fd = os.open(credentials, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as out:
        json.dump(accounts, out)
accounts = json.loads(credentials.read_text())
fresh = not (runtime / "Saves" / "Accounts" / "accounts.xml").exists()
print(f"Local shard: 127.0.0.1:{args.port}; test credentials: {credentials}", flush=True)
# A first boot uses the stock owner-account prompt; subsequent boots are headless.
with (runtime / "server.log").open("a") as log:
    proc = subprocess.Popen(["mono", "ServUO.exe"] + ([] if fresh else ["-service"]), cwd=runtime,
                            stdin=subprocess.PIPE, stdout=log, stderr=log, text=True)
    if fresh:
        proc.stdin.write("y\narena_admin\n" + accounts["arena_admin"] + "\n")
        proc.stdin.flush()
    try:
        raise SystemExit(proc.wait())
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait(timeout=15)
