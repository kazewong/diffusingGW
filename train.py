import argparse
from jaxtyping import PyTree, Float, Array, PRNGKeyArray
from model.jax.sde_score import ScordBasedSDE, GaussianFourierFeatures
from model.jax.common.Unet import Unet
import jax
import jax.numpy as jnp
import torchvision
import optax
import equinox as eqx
from jax._src.distributed import initialize
from jax.experimental.multihost_utils import process_allgather
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from clearml import Task
class GWDiffusionTrainer:

    def __init__(self,
                model, 
                optimizer, 
                scheduler, 
                args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args

    def train(self):
        pass

    def validate(self):
        pass

    def train(
    model: ScordBasedSDE,
    trainloader: DataLoader,
    testloader: DataLoader,
    key: PRNGKeyArray,
    steps: int = 1000,
    print_every: int = 100,
):

    opt_state = optimizer.init(eqx.filter(sde, eqx.is_array))

    @eqx.filter_jit
    def train_step(
        model: ScordBasedSDE,
        opt_state: PyTree,
        batch: Float[Array, "batch 1 28 28"],
        key: PRNGKeyArray,
        opt_update
    ):
        keys = jax.random.split(key, batch.shape[0])
        single_device_loss = lambda model, batch, key: jnp.mean(jax.vmap(model.loss)(batch, key))
        loss_values, grads = eqx.filter_value_and_grad(single_device_loss)(model, batch, keys)
        updates, opt_state = opt_update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_values

    def train_epoch(
        model: ScordBasedSDE,
        opt_state: PyTree,
        trainloader: DataLoader,
        key: PRNGKeyArray,
        epoch: int,
        log_loss: bool = False,
    ):
        train_sampler.set_epoch(epoch)
        train_loss = 0
        for batch in trainloader:
            key, subkey = jax.random.split(key)
            local_batch = jnp.array(batch[0])
            global_shape = (jax.process_count() * local_batch.shape[0], ) + (1,28,28)

            arrays = jax.device_put(jnp.split(local_batch, len(global_mesh.local_devices), axis = 0), global_mesh.local_devices)
            global_batch = jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)
            model, opt_state, loss_values = train_step(model, opt_state, global_batch, subkey, optimizer.update)
            if log_loss: train_loss += jnp.sum(process_allgather(loss_values))
            train_loss = train_loss/ jax.process_count()
            return model, opt_state, train_loss

    @eqx.filter_jit
    def test_step(
        model: ScordBasedSDE,
        batch: Float[Array, "batch 1 28 28"],
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, batch.shape[0])
        loss_values = jnp.mean(jax.vmap(model.loss)(batch, keys))
        return loss_values

    def test_epoch(
        model: ScordBasedSDE,
        testloader: DataLoader,
        key: PRNGKeyArray,
        epoch: int,
    ):
        test_loss = 0
        test_sampler.set_epoch(epoch)
        for batch in testloader:
            key, subkey = jax.random.split(key)
            local_batch = jnp.array(batch[0])
            global_shape = (jax.process_count() * local_batch.shape[0], ) + (1,28,28)

            arrays = jax.device_put(jnp.split(local_batch, len(global_mesh.local_devices), axis = 0), global_mesh.local_devices)
            global_batch = jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)
            test_loss += jnp.sum(process_allgather(test_step(model, global_batch, subkey)))
        test_loss_values = test_loss/ jax.process_count()
        return test_loss_values    


    
    devices = np.array(jax.devices())
    global_mesh = jax.sharding.Mesh(devices, ('b'))
    sharding = jax.sharding.NamedSharding(global_mesh, jax.sharding.PartitionSpec(('b'),))

    max_loss = 1e10
    best_model = model
    for step in range(steps):
        if step % print_every != 0:
            key, subkey = jax.random.split(key)
            model, opt_state, train_loss = train_epoch(model, opt_state, trainloader, subkey, step, log_loss=False)
        if step % print_every == 0:
            key, subkey = jax.random.split(key)
            model, opt_state, train_loss = train_epoch(model, opt_state, trainloader, subkey, step, log_loss=True)
            key, subkey = jax.random.split(key)
            test_loss = test_epoch(model, testloader, subkey, step)

            if max_loss > test_loss:
                max_loss = test_loss
                best_model = model
            if jax.process_index() == 0:
                mlflow.log_metric(key="training_loss", value=train_loss, step=step)
                mlflow.log_metric(key="test_loss", value=test_loss, step=step)
                best_model.save_model(mlflow.get_artifact_uri()[7:] + "/best_model")

    return best_model, opt_state