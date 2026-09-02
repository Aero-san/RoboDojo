from XPolicyLab.policy.Pi_05.deploy import eval_one_episode_batch


class _FakeEnvironment:
    def __init__(self):
        self.steps = []
        self.observation_requests = []

    def is_episode_end(self):
        return len(self.steps) >= 3

    def get_running_env_idx_list(self):
        return [] if self.is_episode_end() else [0, 1]

    def get_obs_batch(self, env_idx_list):
        self.observation_requests.append(list(env_idx_list))
        return [{"env_idx": env_idx} for env_idx in env_idx_list]

    def take_action_batch(self, actions, env_idx_list):
        self.steps.append((list(env_idx_list), list(actions)))


class _FakeModelClient:
    def __init__(self):
        self.inference_requests = []
        self.call_count = {0: 0, 1: 0}

    def call(self, func_name, obs=None):
        if func_name in {"reset", "update_obs_batch"}:
            return None
        if func_name != "get_action_batch":
            raise AssertionError(f"unexpected call: {func_name}")
        env_idx_list = list(obs)
        self.inference_requests.append(env_idx_list)
        chunks = []
        for env_idx in env_idx_list:
            generation = self.call_count[env_idx]
            self.call_count[env_idx] += 1
            chunk_size = 2 if env_idx == 0 else 3
            chunks.append([f"env{env_idx}-chunk{generation}-action{index}" for index in range(chunk_size)])
        return chunks


def test_batch_rollout_replans_each_environment_when_its_chunk_ends():
    environment = _FakeEnvironment()
    model_client = _FakeModelClient()

    eval_one_episode_batch(environment, model_client)

    assert model_client.inference_requests == [[0, 1], [0]]
    assert environment.observation_requests == [[0, 1], [0, 1], [0, 1]]
    assert environment.steps[2][1] == ["env0-chunk1-action0", "env1-chunk0-action2"]
