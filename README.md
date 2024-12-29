# big-data-bowl

srun --pty --partition=hpg-dev --time=3:00:00 --nodes=1 --ntasks=1  --mem=20gb --account=hoover-sai --qos=hoover-sai-b bash -i

jupyter-notebook --no-browser --ip=0.0.0.0 --port 8888


try model with no balancing
more features!!!!
lineman heatmap on plays that caused pressure
lineman heatmap on plays that do not cause pressure
filter out plays where the QB holds onto the ball for too long


Thoughts:
- offense formation is almost always shotgun. we need a better way to encode their location
- train model on Weeks 1-7 and test on Weeks 8-9

