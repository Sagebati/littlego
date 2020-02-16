import random
import numpy as np
import ops
from statistics import mean
from time import time


def data_shuffling(states, policies, values):
    temp = list(zip(states, policies, values))
    random.shuffle(temp)
    t_states, t_policies, t_values = zip(*temp)
    t_states, t_policies, t_values = np.array(t_states), np.array(t_policies), np.array(t_values)
    return t_states, t_policies, t_values


def get_batch(states, policies, values, batch_size, len_train):
    if batch_size == len_train:
        batch_states, batch_policies, batch_values = states, policies, values
    else:
        idx = np.random.randint(low=0, high=len_train, size=batch_size)
        batch_states, batch_policies, batch_values = states[idx], policies[idx], values[idx]
        
    return batch_states, batch_policies, batch_values
    
    
def data_splitting(states, policies, values, len_dataset, test_ratio, k):
    data = {}
    b_split = int(len_dataset * test_ratio * k)
    e_split = int((b_split + len_dataset * test_ratio) % len_dataset)
    data["test_states"], data["test_policies"], data["test_values"] = states[b_split:e_split], policies[b_split:e_split], values[
                                                                                              b_split:e_split]
    data["train_states"] = np.concatenate([states[0:b_split], states[e_split:]])
    data["train_policies"] = np.concatenate([policies[0:b_split], policies[e_split:]])
    data["train_values"] = np.concatenate([values[0:b_split], values[e_split:]])

    data["validation_states"] = data["test_states"]
    data["validation_policies"] = data["test_policies"]
    data["validation_values"] = data["test_values"]

    return data


def supervised_training(dataset, board_size, neural_network,
                        epoch=5000,
                        report_frequency=500, #deprecated
                        save_frequency=4000,
                        batch_size=32,
                        data_size=25000,
                        k_fold=0,
                        test_ratio=1 / 30 # From [Silver et al., 2016] Mastering the game of Go with deep neural networks and tree search
                        ):
    # Training parameters
    k = k_fold  # k-fold cross validation
    random.seed(0)

    maxK = int(1 / test_ratio)
    k = min(k, maxK - 1)

    # Load dataset
    print("Data loading")
    npzfile = np.load(dataset)
    t_states = npzfile['states']
    t_policies = npzfile['policies']
    t_values = npzfile['values']
    player_turn = npzfile['player_turn']

    # Subsample dataset if asked
    if data_size > 0:
        # pre-shuffle
        print("(pre-shuffle)")
        t_states, t_policies, t_values = data_shuffling(t_states, t_policies, t_values)

        print("(subsample dataset)", data_size)
        t_states = t_states[:data_size]
        t_policies = t_policies[:data_size]
        t_values = t_values[:data_size]
        player_turn = player_turn[:data_size]
    # (N, 1, 19, 19, 4) | (N, 362) | (N,)

    # Shape states to neural network input shape
    print("Data shaping")
    tt_states = []
    for i in range(len(t_states)):
        planes = ops.add_player_feature_plane(t_states[i], board_size, player_turn[i])
        tt_states.append(planes)
    t_states = np.array(tt_states)
    input_planes = t_states.shape[-1]
    states, policies, values = ops.reshape_data_for_network(t_states, t_policies, t_values, board_size, input_planes)
    # (N, 19, 19, 5) | (N, 362) | (N, 1)

    # Shuffle
    print("Data shuffling")
    states, policies, values = data_shuffling(states, policies, values)

    # Data splitting
    print("Data splitting")
    len_dataset = len(values)
    test_size = int(len_dataset * test_ratio)
    if test_size != 0:
        b_split = int(len_dataset * test_ratio * k)
        e_split = int((b_split + len_dataset * test_ratio) % len_dataset)
        test_states, test_policies, test_values = states[b_split:e_split], policies[b_split:e_split], values[
                                                                                                      b_split:e_split]
        train_states = np.concatenate([states[0:b_split], states[e_split:]])
        train_policies = np.concatenate([policies[0:b_split], policies[e_split:]])
        train_values = np.concatenate([values[0:b_split], values[e_split:]])

        validation_states, validation_policies, validation_values = test_states, test_policies, test_values

        """splitted_data = data_splitting(states, policies, values, len_dataset, test_ratio, k)
        validation_states = splitted_data["validation_states"]
        validation_policies = splitted_data["validation_policies"]
        validation_values = splitted_data["validation_values"]
        test_states, test_policies, test_values = splitted_data["test_states"], splitted_data["test_policies"], splitted_data["test_values"]
        train_states, train_policies, train_values = splitted_data["train_states"], splitted_data["train_policies"], splitted_data["train_values"]"""

    # Training
    print("Training")
    print(train_states.shape)
    print(train_policies.shape)
    print(train_values.shape)

    #neural_network.save_model()
    len_train = train_states.shape[0]
    train_p_acc, train_v_loss, val_p_acc, val_v_loss, losses = [], [], [], [], []
    total_it = 0

    t0 = time()
    for ep in range(epoch):
        batch_loss, batch_p_acc, batch_v_err = [], [], []
        for it in range(len_train // batch_size):
            # Get batch
            batch_states, batch_policies, batch_values = get_batch(train_states, train_policies, train_values, batch_size, len_train)
            idx = np.random.randint(low=0, high=8)
            #t01 = time()
            batch_states, batch_policies, batch_values = ops.data_augmentation(batch_states, batch_policies, batch_values, board_size, input_planes, idx=idx)
            #t11 = time()
            #print("data augmentation %.3g" % (t11 - t01))

            # Train model on this batch
            #t02 = time()
            loss, p_acc, v_err = neural_network.train(batch_states, batch_policies, batch_values, total_it)
            #t12 = time()
            #print("train %.3g" % (t12 - t02))
            total_it += 1
            batch_loss.append(loss)
            batch_p_acc.append(p_acc)
            batch_v_err.append(v_err)
    
        # Print results
        t1 = time()
        p_acc, v_err, loss = mean(batch_p_acc), mean(batch_v_err), mean(batch_loss)
        print(neural_network.get_learning_rate(total_it))
        print("\n#####################")
        print("# Epoch {} / {}: (%.3g sec) \nloss = {}".format(ep, epoch, loss) % (t1 - t0))
        print("##############")
        print("# TRAINING  :\npolicy accuracy = {:.4f}\nvalue  error    = {:.4f}".format(p_acc, v_err))

        if test_size != 0:
            vali_p_acc, vali_v_err, _, _ = neural_network.feed_forward_accuracies(validation_states,
                                                                                    validation_policies,
                                                                                    validation_values,
                                                                                    ep)
            print("##############")
            print("# VALIDATION:\npolicy accuracy = {:.4f}\nvalue  error    = {:.4f}".format(vali_p_acc,
                                                                                             vali_v_err))
            print("#####################")
            val_p_acc.append(vali_p_acc)
            val_v_loss.append(vali_v_err)
            train_p_acc.append(p_acc)
            train_v_loss.append(v_err)
            losses.append(loss)
            np.savez("loss_epoch",
                     t_p_acc=train_p_acc,
                     t_v_loss=train_v_loss,
                     v_p_acc=val_p_acc,
                     v_v_loss=val_v_loss,
                     loss=losses)
        print()

        """if ep % save_frequency == 0:
            neural_network.save_model(False)"""

    print("Optimization Finished!")
    test_p_acc, test_v_err, _, _ = neural_network.feed_forward_accuracies(test_states, test_policies,
                                                                          test_values, 0)
    print("TEST      :\npolicy accuracy = {:.4f}\nvalue  error    = {:.4f}".format(test_p_acc, test_v_err))







#################################################
# OLD
#################################################
def old_supervised_training(dataset, board_size, neural_network,
                        epoch=50000,
                        report_frequency=500,
                        validation_frequency=2000,
                        save_frequency=4000,
                        batch_size=32,
                        data_size=25000,
                        k_fold=0,
                        test_ratio=1 / 30,
                        ):
    # Training parameters
    k = k_fold  # k-fold cross validation
    random.seed(0)

    maxK = int(1 / test_ratio)
    k = min(k, maxK - 1)

    # Load dataset
    print("Data loading")
    npzfile = np.load(dataset)
    t_states = npzfile['states']
    t_policies = npzfile['policies']
    t_values = npzfile['values']
    player_turn = npzfile['player_turn']

    # TODO - remove the next lines to work on the full dataset
    # pre-shuffle
    print("(pre-shuffle)")
    temp = list(zip(t_states, t_policies, t_values))
    random.shuffle(temp)
    t_states, t_policies, t_values = zip(*temp)
    t_states, t_policies, t_values = np.array(t_states), np.array(t_policies), np.array(t_values)

    t_states = t_states[:data_size]
    t_policies = t_policies[:data_size]
    t_values = t_values[:data_size]
    player_turn = player_turn[:data_size]

    # Shape states to neural network input shape
    print("Data shaping")
    tt_states = []
    tt_policies = []
    for i in range(len(t_states)):
        player_feature_plane = np.full((board_size, board_size), player_turn[i])
        t_state = t_states[i]
        player_feature_plane = ops.goban_to_nn_state(player_feature_plane, board_size)
        tt_states.append(np.concatenate([t_state, player_feature_plane], axis=3))
        tt_policies.append(np.reshape(t_policies[i], (1, board_size ** 2 + 1)))
    t_states = np.array(tt_states)
    t_policies = np.array(tt_policies)

    input_planes = t_states.shape[-1]

    # Data augmentation
    print("Data augmentation")
    states, policies, values = [], [], []
    for i in range(len(t_states)):
        state, policy, value = t_states[i], t_policies[i], t_values[i]
        new_states, new_policies = ops.data_augmentation(state, policy, board_size)
        for j in range(len(new_states)):
            states.append(new_states[j])
            policies.append(new_policies[j])
            values.append(value)

    # Shuffle
    print("Data shuffling")
    temp = list(zip(states, policies, values))
    random.shuffle(temp)
    states, policies, values = zip(*temp)
    states, policies, values = np.array(states), np.array(policies), np.array(values)

    states = np.reshape(states, (-1, board_size, board_size, input_planes))
    policies = np.reshape(policies, (-1, board_size ** 2 + 1))
    values = np.reshape(values, (-1, 1))

    # Data splitting
    print("Data splitting")
    len_dataset = len(values)
    test_size = int(len_dataset * test_ratio)
    if test_size != 0:
        b_split = int(len_dataset * test_ratio * k)
        e_split = int((b_split + len_dataset * test_ratio) % len_dataset)
        test_states, test_policies, test_values = states[b_split:e_split], policies[b_split:e_split], values[
                                                                                                      b_split:e_split]
        train_states = np.concatenate([states[0:b_split], states[e_split:]])
        train_policies = np.concatenate([policies[0:b_split], policies[e_split:]])
        train_values = np.concatenate([values[0:b_split], values[e_split:]])

        validation_states, validation_policies, validation_values = test_states, test_policies, test_values

    train_p_loss = []
    train_v_loss = []
    val_p_loss = []
    val_v_loss = []
    # Training
    print("Training")
    neural_network.save_model()
    len_train = train_states.shape[0]
    for i in range(1, epoch):
        # Get batch
        if batch_size == len_train:
            batch_states, batch_policies, batch_values = train_states, train_policies, train_values
        else:
            idx = []
            while len(idx) < batch_size:
                idx.append(np.random.randint(low=0, high=len_train))
            batch_states, batch_policies, batch_values = train_states[idx], train_policies[idx], train_values[idx]

        # Train model on this batch
        loss, p_acc, v_err = neural_network.train(batch_states, batch_policies, batch_values, epoch)

        # Print results
        if i % report_frequency == 0:
            print("\nMinibatch {} : \nloss = {}".format(i, loss))
            print("TRAINING  :\npolicy accuracy = {:.4f}\nvalue  error    = {:.4f}".format(p_acc, v_err))

        if i % validation_frequency == 0:
            if test_size != 0:
                val_p_acc, val_v_err, p_out, v_out = neural_network.feed_forward_accuracies(validation_states,
                                                                                            validation_policies,
                                                                                            validation_values,
                                                                                            epoch)
                print("\nVALIDATION:\npolicy accuracy = {:.4f}\nvalue  error    = {:.4f}".format(val_p_acc,
                                                                                                 val_v_err))
                val_p_loss.append(val_p_acc)
                val_v_loss.append(val_v_err)
                train_p_loss.append(p_acc)
                train_v_loss.append(v_err)
                np.savez("loss_epoch",
                         t_p_loss=train_p_loss,
                         t_v_loss=train_v_loss,
                         v_p_loss=val_p_loss,
                         v_v_loss=val_v_loss)
            print()

        if i % save_frequency == 0:
            neural_network.save_model(False)

    print("Optimization Finished!")
    test_p_acc, test_v_err, _, _ = neural_network.feed_forward_accuracies(test_states, test_policies,
                                                                          test_values, 0)
    print("TEST      :\npolicy accuracy = {:.4f}\nvalue  error    = {:.4f}".format(test_p_acc, test_v_err))
