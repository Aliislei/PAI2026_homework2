"""モデル、環境、ロールアウトの共通処理。

ワーカーとして実行する場合:
    python rollout.py <ワーカー番号>             対策なしの全ロールアウト
    python rollout.py <ワーカー番号> <発動距離 m>  対策ありの対象ロールアウト
"""
import json
import pathlib
import random
import sys
import zlib

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.libero import get_task_init_states
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

MODEL_ID = "lerobot/pi05_libero_finetuned_v044"
SUITES = ["libero_spatial", "libero_object"]
EPISODES_PER_TASK = 50
REPLAN_STEPS = 5
SEED = 7
NUM_WORKERS = 8

# 環境の設定。モデルの入力解像度に合わせる
IMAGE_SIZE = 256
SETTLE_STEPS = 10  # 開始直後に物体が落ち着くのを待つステップ数
MAX_STEPS = {"libero_spatial": 280, "libero_object": 280}
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]  # 動かず、グリッパーは開いたまま

# 分析の設定
ERROR_THRESHOLD_M = 0.02  # 異常とみなす下降距離の誤差。暫定値
GAIN_WINDOW_STEPS = 10  # 換算係数を求める窓のステップ数
FREE_GAIN = 0.0119  # 接触のない区間で実測した換算係数 [m/アクション]
MIN_FREEZE_DESCENT_M = 0.01  # これより浅い下降では発動しない
FREEZE_TAIL_STEPS = 10  # アプローチ完了後も再推論を止めるステップ数。把持が閉じ切るまで
REACH_RADIUS_M = 0.06  # 対象物に降りたとみなす水平距離。暫定値
REACH_HEIGHT_M = 0.15  # 対象物に降りたとみなす高さの差。暫定値
TOP_N = 5  # 繰り返し推論を行うロールアウト数
REPEATS = 16  # 1 つの状態に対する繰り返し推論の回数

# データの保存先
LOCAL_DIR = pathlib.Path("/content/results")
EPISODE_DIR = LOCAL_DIR / "episodes"
ARCHIVE = pathlib.Path("/content/results.tar.gz")
REMOTE_ARCHIVE = "remote:pai_homework2/approach_analysis/results.tar.gz"
RCLONE_CONF = "/content/rclone.conf"
EPISODE_DIR.mkdir(parents=True, exist_ok=True)

suite_tasks = {name: benchmark.get_benchmark_dict()[name]() for name in SUITES}
policy, config, preprocess, postprocess = None, None, None, None


def load_policy():
    """モデルと前処理、後処理を読み込む。"""
    global policy, config, preprocess, postprocess
    config = PreTrainedConfig.from_pretrained(MODEL_ID)
    config.compile_model = False  # コンパイルは起動に時間がかかり、検証には不要
    policy = PI05Policy.from_pretrained(MODEL_ID, config=config).eval()
    preprocess, postprocess = make_pre_post_processors(config, pretrained_path=MODEL_ID)


def rollout_seed(*key):
    """キーから、実行順に依存しない 32 ビットの種を作る。"""
    return zlib.crc32("/".join(map(str, (SEED, *key))).encode())


def make_env(suite_name, task_id):
    """タスクの環境、初期状態、指示文、把持対象の名前を返す。"""
    suite = suite_tasks[suite_name]
    task = suite.get_task(task_id)
    bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl_file), camera_heights=IMAGE_SIZE, camera_widths=IMAGE_SIZE)
    # 対象の 2 スイートでは、ゴール条件の第 1 引数が把持対象になる。例: (On akita_black_bowl_1 plate_1)
    target = env.env.parsed_problem["goal_state"][0][1]
    return env, get_task_init_states(suite, task_id), task.language, target


def quat_to_axis_angle(quat):
    """(x, y, z, w) のクォータニオンを軸角に変換する。"""
    w = np.clip(quat[3], -1.0, 1.0)
    sin_half = np.sqrt(1.0 - w * w)
    if sin_half < 1e-10:
        return np.zeros(3)
    return quat[:3] / sin_half * 2.0 * np.arccos(w)


def image_tensor(image):
    """学習データの向きに合わせて 180 度回転し、[0, 1] の (1, C, H, W) に変換する。"""
    return torch.from_numpy(image[::-1, ::-1].copy()).permute(2, 0, 1)[None].float() / 255


def infer_chunk(obs, prompt, noise_seed):
    """観測から、正規化を解除したアクションチャンク (chunk_size, 7) を返す。"""
    state = np.concatenate([obs["robot0_eef_pos"], quat_to_axis_angle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]])
    batch = {
        "observation.images.image": image_tensor(obs["agentview_image"]),
        "observation.images.image2": image_tensor(obs["robot0_eye_in_hand_image"]),
        "observation.state": torch.from_numpy(state)[None].float(),
        "task": [prompt],
    }
    generator = torch.Generator().manual_seed(noise_seed)
    noise = torch.randn(1, config.chunk_size, config.max_action_dim, generator=generator).to(config.device)
    with torch.inference_mode():
        chunk = postprocess(policy.predict_action_chunk(preprocess(batch), noise=noise))
    return chunk[0].float().cpu().numpy()


def run_episode(env, init_states, prompt, target, suite_name, task_id, episode, stop_at=None, freeze_depth=None):
    """1 エピソードを実行し、記録を返す。

    stop_at を指定すると、そのステップの観測を返す。freeze_depth を指定すると対策ありで実行する。
    """
    # ロールアウトごとに種を蒔き直す
    seed = rollout_seed(suite_name, task_id, episode)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)

    env.reset()
    obs = env.set_init_state(init_states[episode % len(init_states)])
    for _ in range(SETTLE_STEPS):
        obs, _, _, _ = env.step(DUMMY_ACTION)

    sim = env.env.sim
    target_body = env.env.obj_body_id[target]
    log = {key: [] for key in ["infer_t", "infer_state", "chunks", "step_action", "step_eef", "step_obj"]}
    success = False
    next_infer, chunk_start, freeze_t = 0, 0, -1

    for t in range(MAX_STEPS[suite_name]):
        if t == stop_at:
            return obs
        if t >= next_infer:
            chunk = infer_chunk(obs, prompt, rollout_seed(suite_name, task_id, episode, t, 0))
            log["infer_t"].append(t)
            log["infer_state"].append(env.get_sim_state())
            log["chunks"].append(chunk)
            chunk_start = t
            next_infer = t + REPLAN_STEPS
            # 対策は 1 エピソードに 1 回だけ。配置先への下降では発動させない
            if freeze_depth is not None and freeze_t < 0:
                interval, freeze = next_interval(chunk, freeze_depth)
                next_infer = t + interval
                freeze_t = t if freeze else -1

        # 環境の入力範囲に収める
        action = np.clip(chunk[t - chunk_start], -1.0, 1.0)
        log["step_action"].append(action)
        log["step_eef"].append(obs["robot0_eef_pos"].copy())
        log["step_obj"].append(sim.data.body_xpos[target_body].copy())

        obs, _, done, _ = env.step(action)
        if done:
            success = True
            break

    return {key: np.asarray(value) for key, value in log.items()} | {"success": success, "freeze_t": freeze_t}


def approach_end(actions):
    """閉指令の開始後に z 指令が下降でなくなるインデックスを返す。到達しなければ None を返す。"""
    closing = np.flatnonzero(actions[:, 6] > 0)
    if len(closing) == 0:
        return None
    close = closing[0]
    not_descending = np.flatnonzero(actions[close:, 2] >= 0)
    return int(close + not_descending[0]) if len(not_descending) else None


def next_interval(chunk, freeze_depth):
    """次の推論までのステップ数と、再推論を止めるかどうかを返す。

    予定軌道の最下点までの距離で判定するので、対象物の位置は使わない。
    """
    motion = np.cumsum(np.clip(chunk[:, :3], -1.0, 1.0), axis=0) * FREE_GAIN
    low = int(np.argmin(motion[:, 2]))
    depth = -motion[low, 2]
    horizontal = np.linalg.norm(motion[low, :2])

    # 上昇中、グリッパーを閉じた後、横移動が主の軌道では発動しない
    if motion[REPLAN_STEPS - 1, 2] >= 0 or chunk[0, 6] > 0 or depth < MIN_FREEZE_DESCENT_M or depth < horizontal:
        return REPLAN_STEPS, False

    if depth < freeze_depth:
        end = approach_end(chunk)
        if end is not None and end + FREEZE_TAIL_STEPS <= len(chunk):
            return end + FREEZE_TAIL_STEPS, True
        return REPLAN_STEPS, False

    # 発動距離を次の間隔の途中で横切るなら、推論を前倒しする。悪い距離帯の計画を凍結しないため
    remaining = motion[:REPLAN_STEPS, 2] - motion[low, 2]
    crossing = np.flatnonzero(remaining[1:] <= freeze_depth)
    return (int(crossing[0]) + 1, False) if len(crossing) else (REPLAN_STEPS, False)


def episode_path(suite_name, task_id, episode, directory=EPISODE_DIR):
    return directory / f"{suite_name}_task{task_id:02d}_ep{episode:03d}.npz"


def all_rollouts():
    return [
        (suite_name, task_id, episode)
        for suite_name in SUITES
        for task_id in range(suite_tasks[suite_name].n_tasks)
        for episode in range(EPISODES_PER_TASK)
    ]


def atomic_dir(freeze_depth):
    return pathlib.Path("/content/atomic") / f"d{round(freeze_depth * 100):02d}"


def target_rollouts():
    """対策を試すロールアウトを、ファイルから読み込む。"""
    rollouts = json.loads(pathlib.Path("/content/atomic/targets.json").read_text())
    return sorted(tuple(rollout) for rollout in rollouts)


def run_worker(worker_id, freeze_depth=None):
    """自分の担当分のロールアウトを実行する。保存済みのものは飛ばす。"""
    load_policy()
    env, env_task = None, None
    rollouts = all_rollouts() if freeze_depth is None else target_rollouts()
    directory = EPISODE_DIR if freeze_depth is None else atomic_dir(freeze_depth)
    directory.mkdir(parents=True, exist_ok=True)
    for suite_name, task_id, episode in rollouts[worker_id::NUM_WORKERS]:
        path = episode_path(suite_name, task_id, episode, directory)
        if path.exists():
            continue
        # 同じタスクの間は環境を使い回す
        if env_task != (suite_name, task_id):
            if env is not None:
                env.close()
            env, init_states, prompt, target = make_env(suite_name, task_id)
            env_task = (suite_name, task_id)
        result = run_episode(env, init_states, prompt, target, suite_name, task_id, episode, freeze_depth=freeze_depth)
        # 同期が書きかけのファイルを拾わないよう、書き終えてから名前を変える
        temp_path = path.with_suffix(".tmp")
        with open(temp_path, "wb") as file:
            np.savez_compressed(file, suite=suite_name, task_id=task_id, episode=episode, target=target, **result)
        temp_path.rename(path)
        print(path.name, "success" if result["success"] else "failure", flush=True)


if __name__ == "__main__":
    run_worker(int(sys.argv[1]), float(sys.argv[2]) if len(sys.argv) > 2 else None)
