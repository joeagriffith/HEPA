from glob import glob
import tensorboard_reducer as tbr
import os

verbose = True
datasets = ["MNIST", "ModelNet10"]
for dataset in datasets:
    log_path = f"out/{dataset}/logs/raw/"
    for experiment in os.listdir(log_path):
        experiment_path = log_path + experiment + "/"
        for sub_experiment in os.listdir(experiment_path):
            sub_experiment_path = experiment_path + sub_experiment + "/"
            for trial in os.listdir(sub_experiment_path):
                path = sub_experiment_path + trial + "/"
                tb_events_out_dir = f"out/{dataset}/logs/agg/{experiment}/{sub_experiment}/{trial}/"
                csv_out_dir = f"out/{dataset}/logs/agg/{experiment}/{sub_experiment}/{trial}/reduction.csv"
                if os.path.exists(csv_out_dir):
                    print(f"Skipping {csv_out_dir} because it already exists")
                    continue

                input_event_dirs = sorted(glob(path + "*"))
                joined_dirs = "\n".join(input_event_dirs)
                print(f"Found {len(input_event_dirs)} event dirs:\n{joined_dirs}")

                print(f'dataset is {dataset}')
                reduce_ops = ("mean", "min", "max", "std")

                events_dict = tbr.load_tb_events(input_event_dirs, verbose=verbose)
                reduced_events = tbr.reduce_events(events_dict, reduce_ops, verbose=verbose)
                tbr.write_tb_events(reduced_events, tb_events_out_dir, overwrite=False, verbose=verbose)
                tbr.write_data_file(reduced_events, csv_out_dir, overwrite=False, verbose=verbose)

                print("Reduction complete")