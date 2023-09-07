from ray.rllib.agents.dqn.dqn_policy import DQNTFPolicy
from ray.tune.registry import register_env
from flow.utils.registry import make_create_env
from flow.envs.multiagent import UAVEnvAVARS

from flow.core.params import VehicleParams
from flow.core.params import NetParams
from flow.core.params import EnvParams
from flow.core.params import SumoParams
from flow.core.params import TrafficLightParams

from flow.networks import UAVNetwork
import os

ABS_DIR = os.path.abspath(os.path.dirname(__file__)).split('flow')[0]

# Experiment parameters
N_ROLLOUTS = 18  # number of rollouts per training iteration
N_CPUS = 18  # number of parallel workers
HORIZON = 2700  # time horizon of a single rollout

UAV_INTERSECTIONS = \
    ['659784', '389279', 'cluster_26868380_305313534', 'cluster_389280_434149497', '12639664', '389357']

vehicles = VehicleParams()

# if traffic is in osm, activate this
tl_logic = TrafficLightParams(baseline=False)

flow_params = dict(
    exp_tag='AVARS_dqn',

    env_name=UAVEnvAVARS,

    network=UAVNetwork,

    simulator='traci',

    sim=SumoParams(
        render=False,
        sim_step=1,
        restart_instance=True,
        emission_path="{}/output/AVARS_dqn".format(ABS_DIR)
    ),

    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=900,
        additional_params={
            "controlled_intersections": UAV_INTERSECTIONS,
        },
    ),

    net=NetParams(
        template={
            "net": "{}/scenarios/UAV/center10_closing.net.xml".format(ABS_DIR),
            "rou": "{}/scenarios/UAV/center10_clip_rand.rou.xml".format(ABS_DIR),
            "vtype": "{}/scenarios/UAV/flow_vtypes.add.xml".format(ABS_DIR),
        },
        additional_params={
            "controlled_intersections": UAV_INTERSECTIONS,
        }
    ),

    veh=vehicles,

    tls=tl_logic  # if traffic is in osm, activate this
)

create_env, env_name = make_create_env(params=flow_params, version=0)

# Register as rllib env
register_env(env_name, create_env)

test_env = create_env()
obs_space = test_env.observation_space
act_space = test_env.action_space


def gen_policy():
    """Generate a policy in RLlib."""
    return DQNTFPolicy, obs_space, act_space, {}


# Setup PG with a single policy graph for all agents
POLICY_GRAPHS = {'uav': gen_policy()}


def policy_mapping_fn(_):
    """Map a policy in RLlib."""
    return 'uav'


policies_to_train = ['uav']
