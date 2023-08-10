import collections
import json
import time
import copy
import os
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

from domainbed.datasets import get_dataset, split_dataset
from domainbed import algorithms
from domainbed.evaluator import Evaluator
from domainbed.lr_scheduler import get_scheduler
from domainbed.lib import misc
from domainbed.lib import swa_utils
from domainbed.lib.query import Q
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed import swad as swad_module
from domainbed.scripts.fourier import Batch_Fourier,save_image_from_tensor_batch
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def json_handler(v):
    if isinstance(v, (Path, range)):
        return str(v)
    raise TypeError(f"`{type(v)}` is not JSON Serializable")


def train(test_envs, args, hparams, n_steps, checkpoint_freq, logger, writer, target_env=None):
    logger.info("")

    #######################################################
    # setup dataset & loader
    #######################################################
    args.real_test_envs = test_envs
    hparams.device = device# for log
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    dataset, in_splits, out_splits = get_dataset(test_envs, args, hparams, algorithm_class)
    test_splits = []
    if hparams.indomain_test > 0.0:
        logger.info("!!! In-domain test mode On !!!")
        assert hparams["val_augment"] is False, (
            "indomain_test split the val set into val/test sets. "
            "Therefore, the val set should be not augmented."
        )
        val_splits = []
        for env_i, (out_split, _weights) in enumerate(out_splits):
            n = len(out_split) // 2
            seed = misc.seed_hash(args.trial_seed, env_i)
            val_split, test_split = split_dataset(out_split, n, seed=seed)
            val_splits.append((val_split, None))
            test_splits.append((test_split, None))
            logger.info(
                "env %d: out (#%d) -> val (#%d) / test (#%d)"
                % (env_i, len(out_split), len(val_split), len(test_split))
            )
        out_splits = val_splits

    if target_env is not None:
        testenv_name = f"te_{dataset.environments[target_env]}"
        logger.info(f"Target env = {target_env}")
    else:
        testenv_properties = [str(dataset.environments[i]) for i in test_envs]
        testenv_name = "te_" + "_".join(testenv_properties)

    logger.info(
        "Testenv name escaping {} -> {}".format(testenv_name, testenv_name.replace(".", ""))
    )
    testenv_name = testenv_name.replace(".", "")
    logger.info(f"Test envs = {test_envs}, name = {testenv_name}")

    n_envs = len(dataset)
    train_envs = sorted(set(range(n_envs)) - set(test_envs)) # 训练的环境
    iterator = misc.SplitIterator(test_envs)
    batch_sizes = np.full([n_envs], hparams["batch_size"], dtype=np.int)

    batch_sizes[test_envs] = 0
    batch_sizes = batch_sizes.tolist()

    logger.info(f"Batch sizes for each domain: {batch_sizes} (total={sum(batch_sizes)})")

    # calculate steps per epoch
    steps_per_epochs = [
        len(env) / batch_size
        for (env, _), batch_size in iterator.train(zip(in_splits, batch_sizes))
    ]
    steps_per_epoch = min(steps_per_epochs) # 每个domain都是32个batchsize，并且epoch取最小的那个长度
    # epoch is computed by steps_per_epoch
    prt_steps = ", ".join([f"{step:.2f}" for step in steps_per_epochs])
    logger.info(f"steps-per-epoch for each domain: {prt_steps} -> min = {steps_per_epoch:.2f}")

    # setup loaders # 数据加载器
    
    train_loaders = [
        InfiniteDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=batch_size,
            num_workers=dataset.N_WORKERS
        )
        for (env, env_weights), batch_size in iterator.train(zip(in_splits, batch_sizes))
    ]

    # setup eval loaders
    eval_loaders_kwargs = []
    for i, (env, _) in enumerate(in_splits + out_splits + test_splits):
        batchsize = hparams["test_batchsize"]
        loader_kwargs = {"dataset": env, "batch_size": batchsize, "num_workers": dataset.N_WORKERS}
        if args.prebuild_loader:
            loader_kwargs = FastDataLoader(**loader_kwargs)
        eval_loaders_kwargs.append(loader_kwargs)

    eval_weights = [None for _, weights in (in_splits + out_splits + test_splits)]
    eval_loader_names = ["env{}_in".format(i) for i in range(len(in_splits))]
    eval_loader_names += ["env{}_out".format(i) for i in range(len(out_splits))]
    eval_loader_names += ["env{}_inTE".format(i) for i in range(len(test_splits))]
    eval_meta = list(zip(eval_loader_names, eval_loaders_kwargs, eval_weights))

    #######################################################
    # setup algorithm (model)
    #######################################################
    algorithm = algorithm_class(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(test_envs),
        hparams
    )

    algorithm.to(device)

    n_params = sum([p.numel() for p in algorithm.parameters()])
    logger.info("# of params = %d" % n_params)

    # setup scheduler
    scheduler = get_scheduler(
        hparams["scheduler"],
        algorithm.optimizer,
        hparams["lr"],
        n_steps,
    )

    train_minibatches_iterator = zip(*train_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    #######################################################
    # start training loop
    #######################################################
    evaluator = Evaluator(
        test_envs,
        eval_meta,
        n_envs,
        logger,
        evalmode=args.evalmode,
        debug=args.debug,
        target_env=target_env,
    )

    swad = None
#     hparams["swad"] = False
    if hparams["swad"]:
        swad_algorithm = swa_utils.AveragedModel(algorithm)
        swad_cls = getattr(swad_module, hparams["swad"])
        swad = swad_cls(evaluator, **hparams.swad_kwargs)

    last_results_keys = None
    records = []
    epochs_path = args.out_dir / "results.jsonl"
    
    if args.algorithm=="ACE":
        class_specific_train_dataset = []
        for i in range(dataset.num_classes):
            class_specific_train_dataset.append([])
        print('          num classes :     {}'.format(dataset.num_classes))

        for domain_index in range(4):
            if domain_index not in args.real_test_envs:
                print('domain{}'.format(domain_index))
                for sample_index in range(len(in_splits[domain_index][0])):
                    # print('sample{}'.format(sample_index))
                    data = in_splits[domain_index][0][sample_index]["x"]
                    target = in_splits[domain_index][0][sample_index]["y"]
                    class_specific_train_dataset[target].append([data, target])
        
    
    for step in range(n_steps):
        step_start_time = time.time()
        # batches_dictlist: [{env0_data_key: tensor, env0_...}, env1_..., ...]
        batches_dictlist = next(train_minibatches_iterator)
        # batches: {data_key: [env0_tensor, ...], ...},such as {"y":[tensor1,tensor2,tensor3],"x":[tensor1,tensor2,tensor3]}
        batches = misc.merge_dictlist(batches_dictlist)
        # to device,
        batches = {
            key: [tensor.to(device) for tensor in tensorlist] for key, tensorlist in batches.items()
        }
        if hparams["Fourier"]:
            batches = Batch_Fourier(batches,device,args.lam_alpha,args.lam_dis,args.times)
            images = torch.cat(batches["x"], dim=0)
            save_image_from_tensor_batch(images, 32, path='./batch.jpg', device=device)# check image
            
        if args.algorithm=="ACE":
            minibatches_device = [(batches["x"][i],batches["y"][i]) for i in range(len(batches["y"]))]
            
            neighbour_list = []
            distant_list = []
            label_dist_list = []

            anchor_data = minibatches_device[0][0]
            anchor_label = minibatches_device[0][1]
            bs = anchor_data.shape[0]
            for ibatch in range(bs):
                neighbour_label = anchor_label[ibatch]
                neighbour_idx = np.random.randint( len(class_specific_train_dataset[neighbour_label]) , size=1)[0]
                neighbour = class_specific_train_dataset[neighbour_label][neighbour_idx][0]
                distant_label = np.random.randint(dataset.num_classes, size=1)[0]
                while distant_label == anchor_label[ibatch]:
                    distant_label = np.random.randint(dataset.num_classes, size=1)[0]
                distant_idx = np.random.randint( len(class_specific_train_dataset[distant_label]) , size=1)[0]
                distant = class_specific_train_dataset[distant_label][distant_idx][0]

                label_dist_list.append(distant_label)
                neighbour_list.append(neighbour)
                distant_list.append(distant)

            neighbour_batch   = torch.stack(neighbour_list,0).to(device)
            distant_batch     = torch.stack(distant_list,0).to(device)
            label_dist_list   = torch.tensor(label_dist_list).to(device)

            triplet_minibatches_device = []
            for i in range(bs):
                triplet_minibatches_device.append([anchor_data[i].to(device),torch.tensor([anchor_label[i]]).to(device)])
            for i in range(bs):
                triplet_minibatches_device.append([neighbour_batch[i].to(device),torch.tensor([anchor_label[i]]).to(device)])
            for i in range(bs):
                triplet_minibatches_device.append([distant_batch[i].to(device),torch.tensor([label_dist_list[i]]).to(device)])
            step_vals = algorithm.update(triplet_minibatches_device)
        else:
            inputs = {**batches, "step": step}
            step_vals = algorithm.update(**inputs)#算法更新
            
        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)
        checkpoint_vals["step_time"].append(time.time() - step_start_time)
        
        # swad = None
        if swad:
            # swad_algorithm is segment_swa for swad
            swad_algorithm.update_parameters(algorithm, step=step)

        scheduler.step()

        if step % checkpoint_freq == 0:
            results = {
                "step": step,
                "epoch": step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            eval_start_time = time.time()
            accuracies, summaries = evaluator.evaluate(algorithm)
            results["eval_time"] = time.time() - eval_start_time

            # results = (epochs, loss, step, step_time)
            results_keys = list(summaries.keys()) + sorted(accuracies.keys()) + list(results.keys())
            # merge results
            results.update(summaries)
            results.update(accuracies)

            # print
            if results_keys != last_results_keys:
                logger.info(misc.to_row(results_keys))
                last_results_keys = results_keys
            logger.info(misc.to_row([results[key] for key in results_keys]))
            records.append(copy.deepcopy(results))

            # update results to record
            results.update({"hparams": dict(hparams), "args": vars(args)})

            with open(epochs_path, "a") as f:
                f.write(json.dumps(results, sort_keys=True, default=json_handler) + "\n")

            checkpoint_vals = collections.defaultdict(lambda: [])

            writer.add_scalars_with_prefix(summaries, step, f"{testenv_name}/summary/")
            writer.add_scalars_with_prefix(accuracies, step, f"{testenv_name}/all/")

            if args.model_save and step >= args.model_save:
                ckpt_dir = args.out_dir / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True)

                test_env_str = ",".join(map(str, test_envs))
                filename = "TE{}_{}.pth".format(test_env_str, step)
                if len(test_envs) > 1 and target_env is not None:
                    train_env_str = ",".join(map(str, train_envs))
                    filename = f"TE{target_env}_TR{train_env_str}_{step}.pth"
                path = ckpt_dir / filename

                save_dict = {
                    "args": vars(args),
                    "model_hparams": dict(hparams),
                    "test_envs": test_envs,
                    "model_dict": algorithm.cpu().state_dict(),
                }
                algorithm.cuda()
                if not args.debug:
                    torch.save(save_dict, path)
                else:
                    logger.debug("DEBUG Mode -> no save (org path: %s)" % path)

            # swad
            if swad:
                def prt_results_fn(results, avgmodel):
                    step_str = f" [{avgmodel.start_step}-{avgmodel.end_step}]"
                    row = misc.to_row([results[key] for key in results_keys if key in results])
                    logger.info(row + step_str)

                swad.update_and_evaluate(
                    swad_algorithm, results["train_out"], results["tr_outloss"], prt_results_fn
                )

                if hasattr(swad, "dead_valley") and swad.dead_valley:
                    logger.info("SWAD valley is dead -> early stop !")
                    break

                swad_algorithm = swa_utils.AveragedModel(algorithm)  # reset
        

            
        if step % args.tb_freq == 0:
            # add step values only for tb log
            step_vals["lr"] = scheduler.get_last_lr()[0]
            writer.add_scalars_with_prefix(step_vals, step, f"{testenv_name}/summary/")

    # find best
    logger.info("---")
    records = Q(records)
    oracle_best = records.argmax("test_out")["test_in"]
    iid_best = records.argmax("train_out")["test_in"]
    last = records[-1]["test_in"]

    if hparams.indomain_test:
        # if test set exist, use test set for indomain results
        in_key = "train_inTE"
    else:
        in_key = "train_out"

    iid_best_indomain = records.argmax("train_out")[in_key]
    last_indomain = records[-1][in_key]

    ret = {
        "oracle": oracle_best,
        "iid": iid_best,
        "last": last,
        "last (inD)": last_indomain,
        "iid (inD)": iid_best_indomain,
    }

    # Evaluate SWAD
    if swad:
        swad_algorithm = swad.get_final_model()
        if hparams["freeze_bn"] is False:
            n_steps = 500 if not args.debug else 10
            logger.warning(f"Update SWAD BN statistics for {n_steps} steps ...")
            swa_utils.update_bn(train_minibatches_iterator, swad_algorithm, n_steps)

        logger.warning("Evaluate SWAD ...")
        accuracies, summaries = evaluator.evaluate(swad_algorithm)
        results = {**summaries, **accuracies}
        start = swad_algorithm.start_step
        end = swad_algorithm.end_step
        step_str = f" [{start}-{end}]  (N={swad_algorithm.n_averaged})"
        row = misc.to_row([results[key] for key in results_keys if key in results]) + step_str
        logger.info(row)

        ret["SWAD"] = results["test_in"]
        ret["SWAD (inD)"] = results[in_key]

        ckpt_dir = args.out_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)

        test_env_str = ",".join(map(str, test_envs))
        filename = "TE{}_{}.pth".format(test_env_str, step)
        if len(test_envs) > 1 and target_env is not None:
            train_env_str = ",".join(map(str, train_envs))
            filename = f"TE{target_env}_TR{train_env_str}_{step}.pth"
        path = ckpt_dir / filename

        save_dict = {
            "args": vars(args),
            "model_hparams": dict(hparams),
            "test_envs": test_envs,
            "model_dict": swad_algorithm.cpu().state_dict(),
        }
        #algorithm.cuda()
        if not args.debug:
            torch.save(save_dict, path)
        else:
            logger.debug("DEBUG Mode -> no save (org path: %s)" % path)

    for k, acc in ret.items():
        logger.info(f"{k} = {acc:.3%}")

    return ret, records
