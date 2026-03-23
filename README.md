# Thesis Research Project

Master Thesis on designing synthetic Turing networks using Random Matrix Theory and Machine Learning.

All my code and simulations for this research project. 



## Push Changes to GitHub

How to commit my changes in VS Code:

```bash
git status

git add .

git commit -m "your message"

git push origin main

```

## If git push origin main is rejected

This usually means GitHub has newer changes that are not on your computer yet.

```bash
git status

git pull --rebase origin main

git push origin main
```


## Connect to Imperial VPN

How to connect to the VPN in VS Code

Command + P -> Git: clone 

Command + P -> Remote-SSH:Connect to Host

Enter Imperial password

Go to VS Code Terminal:

```bash
cd ~/TuringNetworksThesis

module load Python/3.12.3-GCCcore-13.3.0

source ~/venvs/thesis/bin/activate
```

How to run and submit jobs on HPC:

```bash
python TopologyRanking/3954-lhs-3node.py # just for testing

time python TopologyRanking/3954-lhs-3node.py # test before to check how long it will run (set correct waiting time in run.pbs)

qsub run.pbs # submit job to HPC

qstat -u $USER # check status of job
```
