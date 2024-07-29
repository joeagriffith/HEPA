from glob import glob
import tensorboard_reducer as tbr
import os

examples = ["MNIST", "ModelNet10"]
for example in examples:
    log_path = f"Examples/{example}/out/logs/"
    for experiment in os.listdir(log_path):
        experiment_path = log_path + experiment + "/"
        for sub_experiment in os.listdir(experiment_path):
            sub_experiment_path = experiment_path + sub_experiment + "/"
            for trial in os.listdir(sub_experiment_path):
                path = sub_experiment_path + trial + "/"
                # check if reduction already exists
                if os.path.exists(path + "reduction.csv"):
                    continue

                input_event_dirs = sorted(glob(path + "*"))
                joined_dirs = "\n".join(input_event_dirs)
                print(f"Found {len(input_event_dirs)} event dirs:\n{joined_dirs}")

                csv_out_dir = path + 'reduction.csv'
                overwrite = True
                reduce_ops = ("mean", "min", "max", "std")
                events_dict = tbr.load_tb_events(input_event_dirs, verbose=True)

                reduced_events = tbr.reduce_events(events_dict, reduce_ops, verbose=True)

                tbr.write_data_file(reduced_events, csv_out_dir, overwrite=overwrite, verbose=True)

                print("Reduction complete")