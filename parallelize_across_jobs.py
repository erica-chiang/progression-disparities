import os
import argparse 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="test_data", help="Path of directory with datasets to fit stan model on")
    parser.add_argument('--start_id', type=int, default=0, help="Index of first dataset to fit stan model on")
    parser.add_argument('--num_jobs', type=int, default=1, help="Number of datasets to fit stan model on")
    parser.add_argument('--data_file_substr', type=str, default="visits_{}.pkl", help="Name structure of data files")

    parser.add_argument('--warmup', type=int, default=8000, help="Number of warmup iterations")
    parser.add_argument('--sampling', type=int, default=8000, help="Number of sample iterations")
    parser.add_argument('--num_chains', type=int, default=4, help="Number of chains")
    parser.add_argument('--stan_model', type=str, default="full_model", help="Name of stan model to fit on data")
    parser.add_argument('--slurm_submission_script', type=str, default="script.sub", help="Name of slurm submission script")
    
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    data_file_substr_full = os.path.join("generated_data", args.data_dir, args.data_file_substr)
    stan_dir = os.path.join("stan_output", args.data_dir + "_" + args.stan_model)
    if not os.path.exists("stan_output"):
        os.makedirs("stan_output")
    if not os.path.exists(stan_dir):
        os.makedirs(stan_dir)
    if not os.path.exists("file_paths"):
        os.makedirs("file_paths")

    cmd = "sbatch --requeue {} python3 {} --warmup {} --sampling {} --num_chains {} --job_id {} --data_file_substr {} --data_dir {} --stan_dir {} --stan_model {}"
    
    file_paths_file = os.path.join("file_paths", args.data_dir + "_" + args.stan_model + "_file_paths.csv")
    file_paths_summary = open(file_paths_file, 'w')
    file_paths_summary.write("stan_sample_file,data_file\n")
    
    if args.stan_model == "full_model":
        fit_script = "fit_full_model.py"
    else:
        fit_script = "fit_ablated_model.py"
        
    for i in range(args.start_id, args.start_id + args.num_jobs):
        
        # submit slurm job
        formatted_cmd = cmd.format(args.slurm_submission_script,
                                    fit_script,
                                    args.warmup,
                                    args.sampling,
                                    args.num_chains,
                                    i, 
                                    args.data_file_substr, 
                                    args.data_dir, 
                                    stan_dir,
                                    args.stan_model)
    
        print(formatted_cmd)
        os.system(formatted_cmd)
        
        stan_sample_file = os.path.join(stan_dir, "az_sum_fit_{}.csv".format(i)) 
        data_file = args.data_file_substr.format(i)
        
        file_paths_summary.write(stan_sample_file + ',' + data_file_substr_full.format(i) + '\n')
    
if __name__ == "__main__":
    main()