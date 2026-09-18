import argparse
import subprocess
from pathlib import Path

def run_script(script_name: str, args: list = None):
    print(f"\n{'='*50}\n[*] EXECUTING: {script_name}\n{'='*50}")
    cmd = ["python", str(Path("src/models") / script_name)]
    if args:
        cmd.extend(args)
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="EV Purchase Predictor Pipeline Orchestrator")
    parser.add_argument("--mode", type=str, required=True, choices=["train_all", "distill"], help="Mode to run")
    args = parser.parse_args()

    if args.mode == "train_all":
        print("[*] Launching Full 160-Model Mega-Blend Training Pipeline...")
        run_script("backbone.py")
        run_script("boundary.py")
        run_script("pseudo.py")
        run_script("neural_net.py")
        print("[+] Training Pipeline Complete!")
    elif args.mode == "distill":
        print("[*] Launching Knowledge Distillation...")
        run_script("distiller.py")

if __name__ == "__main__":
    main()
