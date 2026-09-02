def eval_one_episode(TASK_ENV, model_client):

    model_client.call(func_name="reset") # reset policy

    while not TASK_ENV.is_episode_end(): # Check whether the episode ends
        obs = TASK_ENV.get_obs() # Get Observation
        model_client.call(func_name="update_obs", obs=obs)  # Update Observation
        actions = model_client.call(func_name="get_action") # Get Action according to observation chunk

        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)

            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break

            obs = TASK_ENV.get_obs()
            model_client.call(func_name="update_obs", obs=obs)

def eval_one_episode_batch(TASK_ENV, model_client):

    model_client.call(func_name="reset")
    action_queues = {}
    latest_observations = {}

    while not TASK_ENV.is_episode_end(): # Check whether the episode ends
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        if not env_idx_list:
            break

        replan_env_idx_list = [env_idx for env_idx in env_idx_list if not action_queues.get(env_idx)]
        if replan_env_idx_list:
            missing_obs_envs = [
                env_idx for env_idx in replan_env_idx_list if env_idx not in latest_observations
            ]
            if missing_obs_envs:
                fresh_observations = TASK_ENV.get_obs_batch(missing_obs_envs)
                latest_observations.update(
                    zip(missing_obs_envs, fresh_observations, strict=True)
                )
            obs_list = [latest_observations.pop(env_idx) for env_idx in replan_env_idx_list]
            model_client.call(func_name="update_obs_batch", obs=obs_list)
            action_chunks = model_client.call(
                func_name="get_action_batch",
                obs=replan_env_idx_list,
            )
            for env_idx, action_chunk in zip(replan_env_idx_list, action_chunks, strict=True):
                if not action_chunk:
                    raise ValueError(f"Pi0.5 returned an empty action chunk for env {env_idx}.")
                action_queues[env_idx] = list(action_chunk)

        current_action_list = [action_queues[env_idx].pop(0) for env_idx in env_idx_list]
        TASK_ENV.take_action_batch(current_action_list, env_idx_list)

        running = set(TASK_ENV.get_running_env_idx_list())
        action_queues = {
            env_idx: queue
            for env_idx, queue in action_queues.items()
            if env_idx in running
        }
        latest_observations = {
            env_idx: observation
            for env_idx, observation in latest_observations.items()
            if env_idx in running
        }
        if running:
            running_envs = [env_idx for env_idx in env_idx_list if env_idx in running]
            observations = TASK_ENV.get_obs_batch(running_envs)
            latest_observations.update(zip(running_envs, observations, strict=True))
