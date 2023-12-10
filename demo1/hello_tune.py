import ray
from ray import tune
import time,argparse
import os,time


#This is our kernel, the main function that computes thing of interest
def objective(x, a, b):
    return a * (x ** 0.5) + b
    
#The function the Tune API will call to instantiate a trial
def trainable (config):
    a,b = config['a'],config['b']
    # call the evaluation function on different values of x
    for x in range(20):
        score = objective (x, a, b)
        # Send results to Tune
        tune.report(score = score)  



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Ray Tune helloworld example")
    parser.add_argument("--num-samples", type=int, default=50, 
                        help="Number of samples/trials to run")
    parser.add_argument("--max-concurrent-trials", type=int, 
                        default=4, help="Maximum trials to start concurrently. \
                        Will start equal to or less trials depending on the available resources")
    parser.add_argument('--logs-dir', type=str,
                        default=os.path.join('logs',os.getenv("SLURM_JOBID")),
                        help="Create a logs directory within the experiment directory")
    args = parser.parse_args()
    os.makedirs(args.logs_dir,exist_ok=True)
    
    
    #Connect to Ray server
    ray.init(address=os.environ["ip_head"], _node_ip_address=os.environ["head_node_ip"],_redis_password=os.getenv('redis_password'))
    
    #Calling Tune run to execute the trials with some definitions
    
    analysis = tune.run(trainable,
                        num_samples=args.num_samples,
                        resources_per_trial={'cpu':1,
                                             },
                        config={
                            "a": tune.uniform(0, 20),
                            "b": tune.uniform(0, 20),
                            "max_concurrent_trials": args.max_concurrent_trials
                            },
                        verbose=2,
                        local_dir=args.logs_dir)
    #Print best trial
    print("Best config is:", analysis.get_best_config(metric="score", mode="max"))
