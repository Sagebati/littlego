import random

import numpy as np

import ops


def supervised_training(dataset, board_size, neural_network,
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
