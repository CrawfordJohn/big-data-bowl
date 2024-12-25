# big-data-bowl

srun --pty --partition=hpg-dev --time=3:00:00 --nodes=1 --ntasks=1  --mem=20gb --account=hoover-sai --qos=hoover-sai-b bash -i

jupyter-notebook --no-browser --ip=0.0.0.0 --port 8888