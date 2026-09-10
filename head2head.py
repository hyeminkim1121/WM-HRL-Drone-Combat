"""head2head.py — 학습된 WM팀 vs PPO팀 직접 교전 (로컬 CPU).

한 진영은 WM 상위, 다른 진영은 PPO 상위가 지휘 (하위는 공통 룰베이스).
적(enemy)은 external_enemy 모드로 외부 정책이 서브골 지정, _rsa_mirror가 이동/공격만.
관측은 적 시점으로 미러(진영 swap + x 반전). 공정성 위해 양 진영 바꿔가며 측정.

사용:
  python head2head.py --wm paper_results/wm_ff2_N18_M18_s0/wm_v2_step010000.pt \
                      --ppo paper_results/ppo_ff2_N18_M18_s0/ppo_hrl_final.pt \
                      --n_ally_attack 12 --n_ally_defend 6 --n_ally_base 12 \
                      --n_enemy_attack 12 --n_enemy_defend 6 --n_enemy_base 12 --episodes 60
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import config as C
from env.drone_env import DroneEnv
from lower_agent import rulebased_actions
from wm_upper_v2 import WMUpperTrainerV2, set_discrete_subgoals, indices_to_onehot
from ppo_hrl_upper import PPOHRLTrainer


# ── 정책 래퍼 (act(obs)->macro indices) ──
class WMPolicy:
    """학습 코드(_collect_episode)와 동일한 추론 흐름.

    encode_obs(obs, state.h) → feat=(h,z) → actor → step_dynamics(state, action)
    Note: step_dynamics는 길이1 Transformer라 h의 의미가 제한적이지만,
    actor가 이 h에 맞춰 학습되었으므로 동일하게 사용해야 일관된 행동.
    """
    def __init__(self, ckpt, cfg):
        self.tr = WMUpperTrainerV2(cfg=cfg, device="cpu"); self.tr.load(ckpt)
        self.state = None
    def reset(self):
        self.state = self.tr.tssm.encode_obs(torch.zeros(1, self.tr.tssm.cfg.state_dim))
    def act(self, obs):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32)[None]
            self.state = self.tr.tssm.encode_obs(obs_t, self.state.h)
            feat = torch.cat([self.state.h, self.state.z], -1)
            indices = self.tr.actor.get_indices(feat).squeeze(0).cpu().numpy()
            act_onehot = indices_to_onehot(indices, self.tr.n_targets)
            act_t = torch.tensor(act_onehot, dtype=torch.float32)[None]
            self.state = self.tr.tssm.step_dynamics(self.state, act_t)
            return indices

class PPOPolicy:
    def __init__(self, ckpt, cfg):
        self.tr = PPOHRLTrainer(cfg=cfg, device="cpu"); self.tr.load(ckpt)
    def reset(self): pass
    def act(self, obs):
        with torch.no_grad():
            feat = self.tr.encoder(torch.tensor(obs,dtype=torch.float32)[None])
            idx,_,_ = self.tr.actor.act(feat, deterministic=True)
            return idx.squeeze(0).cpu().numpy()


class DreamerV3Policy:
    """DreamerV3 (JAX) 에이전트를 head2head용으로 래핑."""

    def __init__(self, logdir, env_cfg):
        import pathlib
        ROOT = os.path.dirname(os.path.abspath(__file__))
        DV3_ROOT = os.path.join(ROOT, "dreamerv3")
        DV3_PKG  = os.path.join(DV3_ROOT, "dreamerv3")
        sys.path.insert(0, DV3_ROOT)
        sys.path.insert(1, DV3_PKG)

        import elements, embodied
        import ruamel.yaml as yaml
        from dreamerv3.agent import Agent

        # config 복원 (train_dreamerv3.py와 동일)
        configs = yaml.YAML(typ="safe").load(
            (pathlib.Path(DV3_PKG) / "configs.yaml").read_text())
        config = elements.Config(configs["defaults"])
        config = config.update(configs["size12m"])
        config = config.update({
            "batch_length": 8, "report_length": 8,
            "replay_context": 0,
            "logdir": logdir, "seed": 0,
            "jax": {**configs["defaults"]["jax"], "compute_dtype": "float32"},
        })

        # obs/act space (DroneEmbodiedEnv와 동일)
        from train_dreamerv3 import DroneEmbodiedEnv
        emb_env = DroneEmbodiedEnv(env_cfg)
        obs_space = {k: v for k, v in emb_env.obs_space.items()
                     if not k.startswith("log/")}
        act_space = {k: v for k, v in emb_env.act_space.items()
                     if k != "reset"}
        emb_env.close()

        # 에이전트 생성 + 체크포인트 로드
        self.agent = Agent(obs_space, act_space, elements.Config(
            **config.agent, logdir=config.logdir, seed=config.seed,
            jax=config.jax, batch_size=config.batch_size,
            batch_length=config.batch_length,
            replay_context=config.replay_context,
            report_length=config.report_length,
            consec_train=config.consec_train,
            consec_report=config.consec_report,
        ))
        cp = elements.Checkpoint(pathlib.Path(logdir) / "ckpt")
        cp.agent = self.agent
        cp.load_or_save()
        print(f"[DreamerV3] loaded from {logdir}")

        self.n_attackers = env_cfg.get("n_ally_attack", 12)
        self.n_defenders = env_cfg.get("n_ally_defend", 6)
        self.state = None
        self._is_first = True

    def reset(self):
        self.state = self.agent.init_policy(batch_size=1)
        self._is_first = True

    def act(self, obs):
        obs_dict = {
            "vector": obs[None].astype(np.float32),
            "reward": np.array([0.0], dtype=np.float32),
            "is_first": np.array([self._is_first]),
            "is_last": np.array([False]),
            "is_terminal": np.array([False]),
        }
        act, self.state = self.agent.policy(self.state, obs_dict, mode="eval")
        self._is_first = False

        indices = np.zeros(self.n_attackers + self.n_defenders, dtype=np.int64)
        for i in range(self.n_attackers):
            indices[i] = int(act[f"atk_{i}"][0])
        for i in range(self.n_defenders):
            indices[self.n_attackers + i] = int(act[f"def_{i}"][0])
        return indices


# ── 적 시점 미러 관측 (진영 swap + x 반전) ──
def mirror_obs(obs, layout):
    blk={}; i=0
    for name,n,raw in layout:
        blk[name]=obs[i:i+n*raw].reshape(n,raw).copy(); i+=n*raw
    def flip(a):
        a=a.copy(); a[:,0]=1.0-a[:,0]; return a   # x_norm 반전
    out=[flip(blk["enemy_drone"]), flip(blk["enemy_base"]),
         flip(blk["ally_drone"]),  flip(blk["ally_base"])]
    return np.concatenate([b.flatten() for b in out]).astype(np.float32)


# ── 적 매크로 indices → 적 서브골 (set_discrete_subgoals의 적 버전) ──
def set_enemy_subgoals(env, idx):
    # enemy units are in env._enemy_units (list), NOT env._units (ally-only dict)
    eu = {u.unit_id: u for u in env._enemy_units}
    ally_bases = env._ally_bases
    ally_defs  = [env._units.get(f"ally_defend_{i}") for i in range(env.n_ally_defend)]
    ally_defs  = [d for d in ally_defs if d is not None]
    n_ab = env.n_ally_base
    for i in range(env.n_enemy_attack):
        eid=f"enemy_attack_{i}"; u=eu.get(eid)
        if u is None or not u.alive: continue
        t=int(idx[i])
        if t < n_ab:
            b = ally_bases[t] if t < len(ally_bases) else None
            if b is not None and b.alive: env._enemy_subgoals[eid]=(b.x,b.y)
            else:
                al=[x for x in ally_bases if x.alive]
                if al: n=min(al,key=lambda z:abs(z.x-u.x)+abs(z.y-u.y)); env._enemy_subgoals[eid]=(n.x,n.y)
        else:
            di=t-n_ab
            al_d=[d for d in ally_defs if d.alive]
            if di < len(ally_defs) and ally_defs[di].alive:
                d=ally_defs[di]; env._enemy_subgoals[eid]=(d.x,d.y)
            elif al_d:
                n=min(al_d,key=lambda z:abs(z.x-u.x)+abs(z.y-u.y)); env._enemy_subgoals[eid]=(n.x,n.y)
            else:
                al=[x for x in ally_bases if x.alive]
                if al: n=min(al,key=lambda z:abs(z.x-u.x)+abs(z.y-u.y)); env._enemy_subgoals[eid]=(n.x,n.y)
    alive_atk=[env._units.get(f"ally_attack_{k}") for k in range(env.n_ally_attack)]
    alive_atk=[a for a in alive_atk if a is not None and a.alive]
    for i in range(env.n_enemy_defend):
        eid=f"enemy_defend_{i}"; u=eu.get(eid)
        if u is None or not u.alive: continue
        if not alive_atk: continue
        z=int(idx[env.n_enemy_attack+i]) % env.n_enemy_base
        zb=env._enemy_bases[z]
        tgt=min(alive_atk, key=lambda a:abs(a.x-zb.x)+abs(a.y-zb.y))
        env._enemy_atk_targets[eid]=tgt.unit_id
        env._enemy_subgoals[eid]=(tgt.x,tgt.y)


def play(env_cfg, layout, ally_pol, enemy_pol, seed):
    env=DroneEnv(config=env_cfg); env.reset(seed=seed)
    ally_pol.reset(); enemy_pol.reset()
    K=env_cfg["high_level_interval"]; sik=0; last={}
    while env.agents:
        if sik==0:
            obs=env.get_global_state()
            set_discrete_subgoals(env, ally_pol.act(obs))            # 아군 = ally_pol
            set_enemy_subgoals(env, enemy_pol.act(mirror_obs(obs,layout)))  # 적 = enemy_pol
        _,rew,term,trunc,infos=env.step(rulebased_actions(env))
        if infos: last=next(iter(infos.values()))
        sik=(sik+1)%K
        if all(term.values()) or all(trunc.values()) or last.get("won") or last.get("lost"): break
    # 잔존 통계 (타임아웃 시 판정용)
    ally_bases_alive  = sum(1 for b in env._ally_bases if b.alive)
    enemy_bases_alive = sum(1 for b in env._enemy_bases if b.alive)
    ally_base_hp  = sum(b.hp for b in env._ally_bases if b.alive)
    enemy_base_hp = sum(b.hp for b in env._enemy_bases if b.alive)
    ally_drones   = sum(1 for a in env.possible_agents if env._units.get(a) and env._units[a].alive)
    enemy_drones  = sum(1 for u in env._enemy_units if u.alive)
    env.close()
    if last.get("won"):  return "ally"
    if last.get("lost"): return "enemy"
    # 1순위: 적 기지 더 많이 파괴한 쪽
    ally_destroyed  = len(env._enemy_bases) - enemy_bases_alive
    enemy_destroyed = len(env._ally_bases) - ally_bases_alive
    if ally_destroyed > enemy_destroyed: return "ally"
    if enemy_destroyed > ally_destroyed: return "enemy"
    # 2순위: 상대 잔존 기지 HP가 낮은 쪽 (더 많이 깎은 쪽 승리)
    if enemy_base_hp < ally_base_hp: return "ally"
    if ally_base_hp < enemy_base_hp: return "enemy"
    # 3순위: 드론 생존수 (방어 우위 반영)
    if ally_drones > enemy_drones: return "ally"
    if enemy_drones > ally_drones: return "enemy"
    return "draw"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--wm", required=True); p.add_argument("--ppo", default=None)
    p.add_argument("--wm2", default=None, help="WM vs WM 교전 시 상대 WM 체크포인트")
    p.add_argument("--dv3", default=None, help="DreamerV3 logdir (JAX agent.pkl)")
    p.add_argument("--n_ally_attack",type=int,default=5); p.add_argument("--n_ally_defend",type=int,default=3)
    p.add_argument("--n_ally_base",type=int,default=5); p.add_argument("--n_enemy_attack",type=int,default=5)
    p.add_argument("--n_enemy_defend",type=int,default=3); p.add_argument("--n_enemy_base",type=int,default=5)
    p.add_argument("--focus_fire_k",type=int,default=2); p.add_argument("--enemy_power",type=int,default=5)
    p.add_argument("--episodes",type=int,default=60)
    p.add_argument("--model_size",type=str,default="small",choices=["small","dreamer","large"])
    p.add_argument("--pure_tssm", action="store_true", help="WM1(--wm)을 Pure TSSM으로 로드")
    p.add_argument("--pure_tssm2", action="store_true", help="WM2(--wm2)를 Pure TSSM으로 로드")
    a=p.parse_args()
    env_cfg,wm_cfg=C.make_configs(n_ally_attack=a.n_ally_attack,n_ally_defend=a.n_ally_defend,n_ally_base=a.n_ally_base,
        n_enemy_attack=a.n_enemy_attack,n_enemy_defend=a.n_enemy_defend,n_enemy_base=a.n_enemy_base,
        model_size=a.model_size)
    env_cfg=dict(env_cfg); env_cfg["focus_fire_k"]=a.focus_fire_k; env_cfg["enemy_atk_power"]=a.enemy_power
    env_cfg["external_enemy"]=True
    cfg=dict(wm_cfg); cfg["tssm"]=dict(wm_cfg["tssm"])
    layout=wm_cfg["tssm"]["entity_layout"]

    # PPO는 PPO_CFG 기반 dims 필요 → ppo_hrl_upper가 cfg.get으로 읽음
    from ppo_hrl_upper import PPO_CFG
    pcfg=dict(PPO_CFG); pcfg.update({"state_dim":wm_cfg["tssm"]["state_dim"],
        "n_action_agents":wm_cfg["n_action_agents"],"n_action_targets":wm_cfg["n_action_targets"],
        "n_action_attackers":wm_cfg["n_action_attackers"],"n_action_ally_zones":wm_cfg["n_action_ally_zones"]})

    N=wm_cfg["n_action_agents"]; M=wm_cfg["n_action_targets"]

    # WM vs WM 모드 (--wm2) 또는 WM vs PPO 모드 (--ppo) 또는 WM vs DreamerV3 (--dv3)
    def make_wm_cfg(pure):
        c = dict(cfg)
        c["tssm"] = dict(cfg["tssm"])
        if pure:
            c["tssm"]["use_entity_encoder"] = False
        return c

    if a.dv3:
        label_a, label_b = "WM", "DV3"
        print(f"[h2h] N={N} M={M} | WM vs DreamerV3 | episodes={a.episodes} (양진영 각 {a.episodes//2})")
        policy_a = WMPolicy(a.wm, make_wm_cfg(a.pure_tssm))
        policy_b = DreamerV3Policy(a.dv3, dict(env_cfg))
    elif a.wm2:
        label_a, label_b = "WM1", "WM2"
        print(f"[h2h] N={N} M={M} | WM1 vs WM2 | episodes={a.episodes} (양진영 각 {a.episodes//2})")
        policy_a = WMPolicy(a.wm, make_wm_cfg(a.pure_tssm))
        policy_b = WMPolicy(a.wm2, make_wm_cfg(a.pure_tssm2))
    elif a.ppo:
        label_a, label_b = "WM", "PPO"
        print(f"[h2h] N={N} M={M} | WM vs PPO | episodes={a.episodes} (양진영 각 {a.episodes//2})")
        policy_a = WMPolicy(a.wm, make_wm_cfg(a.pure_tssm))
        policy_b = PPOPolicy(a.ppo, pcfg)
    else:
        print("[h2h] --ppo, --wm2, 또는 --dv3 중 하나를 지정하세요")
        sys.exit(1)

    res={"a":0,"b":0,"draw":0}
    half=a.episodes//2
    for e in range(half):
        r=play(env_cfg,layout,policy_a,policy_b,seed=e)
        res["a" if r=="ally" else ("b" if r=="enemy" else "draw")]+=1
    for e in range(half):
        r=play(env_cfg,layout,policy_b,policy_a,seed=1000+e)
        res["b" if r=="ally" else ("a" if r=="enemy" else "draw")]+=1
    tot=res["a"]+res["b"]+res["draw"]
    print(f"[h2h] N={N}xM={M} 교전 결과 ({tot}판): "
          f"{label_a}승 {res['a']} ({res['a']/tot:.2f}) | {label_b}승 {res['b']} ({res['b']/tot:.2f}) | 무 {res['draw']}")

if __name__=="__main__":
    main()
