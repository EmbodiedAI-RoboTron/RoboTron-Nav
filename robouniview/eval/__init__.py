import os

def eval_one_epoch_ddp(args, dataset_path, **kwarg):

    data_types = args.data_eval if hasattr(args, 'data_eval') else args.data_type
    data_types = [data_types] if not isinstance(data_types, list) else data_types
    eval_log_dir_tmp = kwarg.pop('eval_log_dir')
    print(data_types)
    args.eval_loop_num = 20 # 20
    for ith in range(len(data_types)):
        try:
            print(f"evaluation {data_types[ith]}")
            eval_log_dir = os.path.join(eval_log_dir_tmp, data_types[ith])

            if data_types[ith] == 'chores':
                from .eval_with_chores_new import eval_one_epoch_calvin_ddp as eval_one_epoch_chores_ddp
                eval_one_epoch_chores_ddp(args, dataset_path=dataset_path[ith], eval_log_dir=eval_log_dir, **kwarg)
            else:
                # assert False
                print("no data_types[ith] evaluation!!")
        except Exception as e:
            print(e)
            pass
            