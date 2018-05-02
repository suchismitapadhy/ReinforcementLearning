# Instructions to replicate the results
* To run 3.1 use :

    ``python behavioral_cloning.py --max_timesteps 1000 --training_epochs 20001 --check_every 5000 experts/HalfCheetah-v1.pkl HalfCheetah-v1``

    ``python behavioral_cloning.py --max_timesteps 1000 --training_epochs 20001 --check_every 5000 experts/Humanoid-v1.pkl Humanoid-v1``

* To run 3.2 use :
    
    ``python behavioral_cloning.py --max_timesteps 1000 --training_epochs 10001 --check_every 500 experts/HalfCheetah-v1.pkl HalfCheetah-v1``

* To run 4 use : 

    ``python behavioral_cloning.py --run_type dag --dagger_steps 10 --max_timesteps 1000 --training_epochs 5001 --check_every 5000 experts/Humanoid-v1.pkl Humanoid-v1``
