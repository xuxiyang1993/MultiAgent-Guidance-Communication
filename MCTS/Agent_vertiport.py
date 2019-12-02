import argparse
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

import sys

sys.path.extend(['../Simulators'])
from nodes_multi import MultiAircraftNode, MultiAircraftState
from search_multi import MCTS
from config_vertiport import Config
from MultiAircraftVertiportEnv import MultiAircraftEnv


# from testEnv import MultiAircraftEnv

def run_experiment(env, no_episodes, render, save_path, decentralized):
    text_file = open(save_path, "w")  # save all non-terminal print statements in a txt file
    episode = 0
    epi_returns = []
    conflicts_list = []
    num_aircraft = Config.num_aircraft
    time_dict = {}

    while episode < no_episodes:
        # at the beginning of each episode, set done to False, set time step in this episode to 0
        # set reward to 0, reset the environment
        episode += 1
        done = False
        episode_time_step = 0
        episode_reward = 0
        last_observation, id_list = env.reset()
        action_by_id = {}
        info = None
        near_end = False
        counter = 0  # avoid end episode initially

        # before = time.time()
        while not done:
            if render:
                env.render()
                # import ipdb
                # ipdb.set_trace()
            # make decision every 5 time steps

            # if episode_time_step > 1904:
            #     import ipdb; ipdb.set_trace()

            if episode_time_step % 5 == 0:

                time_before = int(round(time.time() * 1000))
                num_existing_aircraft = last_observation.shape[0]
                action = np.ones(num_existing_aircraft, dtype=np.int32)
                action_by_id = {}

                # make decision for each aircraft one by one using centralized controller
                if not decentralized or num_existing_aircraft == 0:
                    for index in range(num_existing_aircraft):
                        if id_list[index] in env.centralized_controller.missing_aircraft:
                            continue
                        state = MultiAircraftState(state=last_observation, index=index, init_action=action)
                        root = MultiAircraftNode(state=state)
                        mcts = MCTS(root)
                        # if aircraft if close to another aircraft, build a larger tree, else build smaller tree

                        # if episode_time_step > 1904:
                        #     if id_list[index] == 79 or id_list[index] == 82:
                        #         import ipdb; ipdb.set_trace()

                        if info[id_list[index]] < 4 * Config.minimum_separation:
                            best_node = mcts.best_action(Config.no_simulations, Config.search_depth)
                        else:
                            best_node = mcts.best_action(Config.no_simulations_lite, Config.search_depth_lite)
                        action[index] = best_node.state.prev_action[index]
                        action_by_id[id_list[index]] = best_node.state.prev_action[index]

                else:
                    # ac_ids = list(env.aircraft_dict.ac_dict.keys())
                    # aircrafts[-1], aircrafts[0] = aircrafts[0], aircrafts[-1]
                    # env.aircraft_dict.ac_dict.move_to_end(ac_ids[-1], last=False)
                    # env.aircraft_dict.ac_dict.move_to_end(ac_ids[0])
                    lost = False
                    i = 0
                    for ac_id, ac in env.aircraft_dict.ac_dict.items():
                        current_ac = ac
                        if lost or i == 0:
                            current_ac = current_ac.get_aircraft_info(env.aircraft_dict.ac_dict)
                            env.aircraft_dict.ac_dict[current_ac.id] = current_ac
                            prev_ac = current_ac
                            i += 1
                            lost = False
                            continue
                        else:
                            with ThreadPoolExecutor(max_workers=2) as pool:
                                decision = pool.submit(prev_ac.make_decision)
                                status = pool.submit(current_ac.get_aircraft_info, env.aircraft_dict.ac_dict)
                            prev_ac = decision.result()
                            current_ac = status.result()
                            env.aircraft_dict.ac_dict[prev_ac.id] = prev_ac
                            env.aircraft_dict.ac_dict[current_ac.id] = current_ac

                        if current_ac.id not in prev_ac.miss_ids:  # no loss
                            prev_ac.broadcast_action(env.aircraft_dict.ac_dict)
                            lost = False
                        else:
                            with ThreadPoolExecutor(max_workers=2) as pool:
                                broadcast = pool.submit(prev_ac.broadcast_action, env.aircraft_dict.ac_dict)
                                decision = pool.submit(current_ac.make_decision)
                            env.aircraft_dict.ac_dict = broadcast.result()
                            env.aircraft_dict.ac_dict[current_ac.id] = decision.result()
                            current_ac.broadcast_action(env.aircraft_dict.ac_dict)
                            lost = True
                        prev_ac = current_ac

                    last_aircraft = list(env.aircraft_dict.ac_dict.values())[-1].make_decision()
                    env.aircraft_dict.ac_dict[last_aircraft.id] = last_aircraft
                    action_by_id = {}
                    for ac in env.aircraft_dict.ac_dict.values():
                        # if ac.action is None:
                        #     print(env.aircraft_dict.ac_dict)
                        #     print(ac.id)
                        action_by_id[ac.id] = ac.action

                    assert len(action_by_id) == num_existing_aircraft
                    # for air in env.aircraft_dict.ac_dict.values():
                    #     try:
                    #         assert action[air.idx] == action_by_id[air.id]
                    #     except AssertionError:
                    #         print(f'ID: {air.id}')
                    #         print(f'Action: {action[air.idx]}')
                    #         print(f'Action_by_id: {action_by_id[air.id]}')
                    #         import ipdb
                    #         ipdb.set_trace()

                time_after = int(round(time.time() * 1000))
                if num_existing_aircraft in time_dict:
                    time_dict[num_existing_aircraft].append(time_after - time_before)
                else:
                    time_dict[num_existing_aircraft] = [time_after - time_before]
            (observation, id_list), reward, done, info = env.step(action_by_id, near_end)

            episode_reward += reward
            last_observation = observation
            episode_time_step += 1

            if episode_time_step % 100 == 0:
                print('========================== Time Step: %d =============================' % episode_time_step,
                      file=text_file)
                print('Number of conflicts:', env.conflicts / 2, file=text_file)
                print('Total Aircraft Genrated:', env.id_tracker, file=text_file)
                print('Goal Aircraft:', env.goals, file=text_file)
                print('NMACs:', env.NMACs / 2, file=text_file)
                print('NMAC/h:', (env.NMACs / 2) / (env.total_timesteps / 3600), file=text_file)
                print('Total Flight Hours:', env.total_timesteps / 3600, file=text_file)
                print('Current Aircraft Enroute:', env.aircraft_dict.num_aircraft, file=text_file)

                print('========================== Time Step: %d =============================' % episode_time_step)
                print('Number of conflicts:', env.conflicts / 2)
                print('Total Aircraft Genrated:', env.id_tracker)
                print('Goal Aircraft:', env.goals)
                print('NMACs:', env.NMACs / 2)
                print('NMAC/h:', (env.NMACs / 2) / (env.total_timesteps / 3600))
                print('Total Flight Hours:', env.total_timesteps / 3600)
                print('Current Aircraft Enroute:', env.aircraft_dict.num_aircraft)
                # after = time.time()
                # print('Current time:', after - before)

            if env.id_tracker - 1 >= 10000:
                counter += 1
                near_end = True

            if counter > 0:
                done = num_existing_aircraft == 0

        print('========================== End =============================', file=text_file)
        print('========================== End =============================')
        print('Number of conflicts:', env.conflicts / 2)
        print('Total Aircraft Genrated:', env.id_tracker)
        print('Goal Aircraft:', env.goals)
        print('NMACs:', env.NMACs / 2)
        print('Current Aircraft Enroute:', env.aircraft_dict.num_aircraft)
        for key, item in time_dict.items():
            print('%d aircraft: %.2f' % (key, np.mean(item)))

        # print training information for each training episode
        epi_returns.append(info)
        conflicts_list.append(env.conflicts)
        print('Training Episode:', episode)
        print('Cumulative Reward:', episode_reward)

    time_list = time_dict.values()
    flat_list = [item for sublist in time_list for item in sublist]
    print('----------------------------------------')
    print('Number of aircraft:', Config.num_aircraft)
    print('Search depth:', Config.search_depth)
    print('Simulations:', Config.no_simulations)
    print('Time:', sum(flat_list) / float(len(flat_list)))
    print('NMAC prob:', epi_returns.count('n') / no_episodes)
    print('Goal prob:', epi_returns.count('g') / no_episodes)
    print('Average Conflicts per episode:',
          sum(conflicts_list) / float(len(conflicts_list)) / 2)  # / 2 to ignore duplication
    env.close()
    text_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_episodes', '-e', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--save_path', '-p', type=str, default='output/seed2.txt')
    parser.add_argument('--debug', '-d', action='store_true')
    parser.add_argument('--render', '-r', action='store_true')
    parser.add_argument('--decentralized', '-dc', action='store_true')
    args = parser.parse_args()

    import random
    np.set_printoptions(suppress=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    env = MultiAircraftEnv(args.seed, args.debug, args.decentralized)
    run_experiment(env, args.no_episodes, args.render, args.save_path, args.decentralized)


if __name__ == '__main__':
    main()
