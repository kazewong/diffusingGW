import argparse
from jaxtyping import PyTree, Float, Array, PRNGKeyArray
from model.jax.diffusion.sde_score import ScordBasedSDE, GaussianFourierFeatures
from model.jax.common.Unet import Unet
import jax
import jax.numpy as jnp
import torchvision
import optax
import equinox as eqx
from jax._src.distributed import initialize
from jax.experimental.multihost_utils import process_allgather
from data.GWdataset import GWdataset
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from clearml import Task, Logger



argparser = argparse.ArgumentParser()

# Metadata about the experiment
argparser.add_argument("--data_path", type=str)
argparser.add_argument("--experiment_name", type=str, default='None')

# Model hyperparameters
argparser.add_argument("--time_feature", type=int, default=128)
argparser.add_argument("--autoencoder_embed_dim", type=int, default=256)
argparser.add_argument('--hidden_layer', nargs='+', type=int, default=[1,16,32,64,128])
argparser.add_argument('--group_norm_size', type=int, default=32)
argparser.add_argument("--sigma", type=float, default=25.0)

# Training hyperparameters
argparser.add_argument("--n_epochs", type=int, default=500)
argparser.add_argument("--batch_size", type=int, default=128)
argparser.add_argument("--learning_rate", type=float, default=1e-4)
argparser.add_argument("--log_epoch", type=int, default=2)
argparser.add_argument("--seed", type=int, default=2019612721831)
argparser.add_argument("--num_workers", type=int, default=8)

args = argparser.parse_args()

initialize()
n_processes = jax.process_count()
if jax.process_index() == 0:
    Task.init(project_name="DiffusionAstro", task_name=args.experiment_name)


class GWDiffusionTrainer:

    def __init__(self,
                config: argparse.Namespace, logging: bool = False):
        self.config = config
        self.logging = logging
        # if logging:
        #     Task.connect_configuration(configuration=config)

        devices = np.array(jax.devices())
        self.global_mesh = jax.sharding.Mesh(devices, ('b'))
        self.sharding = jax.sharding.NamedSharding(self.global_mesh, jax.sharding.PartitionSpec(('b'),))

        train_set, test_set = random_split(GWdataset(config.data_path),[0.8,0.2])
        train_sampler = DistributedSampler(train_set,
                                           num_replicas=n_processes,
                                           rank=jax.process_index(),
                                           shuffle=True,
                                           seed=config.seed)
        test_sampler = DistributedSampler(test_set,
                                            num_replicas=n_processes,
                                            rank=jax.process_index(),
                                            shuffle=False,
                                            seed=config.seed)
        self.train_loader = DataLoader(train_set,
                                        batch_size=config.batch_size,
                                        num_workers=config.num_workers,
                                        sampler=train_sampler,
                                        pin_memory=True)
        self.test_loader = DataLoader(test_set,
                                        batch_size=config.batch_size,
                                        num_workers=config.num_workers,
                                        sampler=test_sampler,
                                        pin_memory=True)

        self.data_shape = train_set.dataset.get_shape()


        self.key, subkey = jax.random.split(jax.random.PRNGKey(config.seed))
        unet = Unet(1, config.hidden_layer, config.autoencoder_embed_dim, subkey, group_norm_size=config.group_norm_size)
        self.key, subkey = jax.random.split(self.key)
        time_embed = eqx.nn.Linear(config.time_feature, config.autoencoder_embed_dim, key=subkey)
        self.key, subkey = jax.random.split(self.key)
        gaussian_feature = GaussianFourierFeatures(config.time_feature, subkey)
        self.model = ScordBasedSDE(unet,
                                    lambda x: 1,
                                    lambda x: 1.0,
                                    lambda x: config.sigma**x,
                                    lambda x: jnp.sqrt((config.sigma**(2 * x) - 1.) / 2. / jnp.log(config.sigma)),
                                    gaussian_feature,
                                    time_embed)

        self.optimizer = optax.adam(config.learning_rate)
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))

    def train(self):
        if jax.process_index()==0: print("Start training")
        max_loss = 1e10
        self.best_model = self.model
        for step in range(self.config.n_epochs):
            if jax.process_index()==0: print("Epoch: ", step)
            if step % self.config.log_epoch == 0:
                self.key, subkey = jax.random.split(self.key)
                self.model, self.opt_state, train_loss = self.train_epoch(self.model, self.opt_state, self.train_loader, subkey, step, log_loss=True)
                self.key, subkey = jax.random.split(self.key)
                test_loss = self.test_epoch(self.model, self.test_loader, subkey, step)

                if max_loss > test_loss:
                    max_loss = test_loss
                    self.best_model = self.model
                if self.logging:
                    Logger.current_logger().report_scalar("Loss", "training_loss", value=train_loss, iteration=step)
                    Logger.current_logger().report_scalar("Loss", "test_loss", value=test_loss, iteration=step)
                    self.best_model.save_model("./best_model")
                    Task.current_task().upload_artifact(artifact_object="./best_model", name="model")
            else:
                self.key, subkey = jax.random.split(self.key)
                self.model, self.opt_state, train_loss = self.train_epoch(self.model, self.opt_state, self.train_loader, subkey, step, log_loss=False)


    def validate(self):
        pass

    @staticmethod
    @eqx.filter_jit
    def train_step(
        model: ScordBasedSDE,
        opt_state: PyTree,
        batch: Float[Array, "batch 1 datashape"],
        key: PRNGKeyArray,
        opt_update
    ):
        keys = jax.random.split(key, batch.shape[0])
        single_device_loss = lambda model, batch, key: jnp.mean(jax.vmap(model.loss)(batch, key))
        loss_values, grads = eqx.filter_value_and_grad(single_device_loss)(model, batch, keys)
        updates, opt_state = opt_update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_values

    @staticmethod
    @eqx.filter_jit
    def test_step(
        model: ScordBasedSDE,
        batch: Float[Array, "batch 1 datashape"],
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, batch.shape[0])
        loss_values = jnp.mean(jax.vmap(model.loss)(batch, keys))
        return loss_values

    def train_epoch(self,
        model: ScordBasedSDE,
        opt_state: PyTree,
        trainloader: DataLoader,
        key: PRNGKeyArray,
        epoch: int,
        log_loss: bool = False,
    ):
        self.train_loader.sampler.set_epoch(epoch)
        train_loss = 0
        for batch in trainloader:
            key, subkey = jax.random.split(key)
            local_batch = jnp.array(batch)
            global_shape = (jax.process_count() * local_batch.shape[0], ) + (1,) + self.data_shape

            arrays = jax.device_put(jnp.split(local_batch, len(self.global_mesh.local_devices), axis = 0), self.global_mesh.local_devices)
            global_batch = jax.make_array_from_single_device_arrays(global_shape, self.sharding, arrays)
            model, opt_state, loss_values = self.train_step(model, opt_state, global_batch, subkey, self.optimizer.update)
            if log_loss: train_loss += jnp.sum(process_allgather(loss_values))
            train_loss = train_loss/ jax.process_count()
            return model, opt_state, train_loss

    def test_epoch(self,
        model: ScordBasedSDE,
        testloader: DataLoader,
        key: PRNGKeyArray,
        epoch: int,
    ):
        test_loss = 0
        self.test_loader.sampler.set_epoch(epoch)
        for batch in testloader:
            key, subkey = jax.random.split(key)
            local_batch = jnp.array(batch)
            global_shape = (jax.process_count() * local_batch.shape[0], ) + (1,) + self.data_shape

            arrays = jax.device_put(jnp.split(local_batch, len(self.global_mesh.local_devices), axis = 0), self.global_mesh.local_devices)
            global_batch = jax.make_array_from_single_device_arrays(global_shape, self.sharding, arrays)
            test_loss += jnp.sum(process_allgather(self.test_step(model, global_batch, subkey)))
        test_loss_values = test_loss/ jax.process_count()
        return test_loss_values    

if jax.process_index() == 0:
    GWDiffusionTrainer(args, logging=True).train()
else:
    GWDiffusionTrainer(args, logging=False).train()

