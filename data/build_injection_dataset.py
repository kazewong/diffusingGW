from jimgw.single_event.detector import H1
from jimgw.single_event.waveform import RippleIMRPhenomD
from jaxtyping import Float, Array, PRNGKeyArray
import jax
import jax.numpy as jnp
import h5py
import numpy as np

duration: Float = 4.0
sampling_rate: int = 4096
f_min: Float = 10.0
freqs = jnp.fft.rfftfreq(int(duration * sampling_rate), 1 / sampling_rate)
freqs = freqs[freqs >= f_min]
psd = H1.load_psd(freqs)
var = psd / (4 * (freqs[1] - freqs[0]))
n_sample = psd.shape[0]

@jax.jit
@jax.vmap
def generate_noise(
    rng: PRNGKeyArray,
) -> Float[Array, " n_sample"]:
    """Generate noise with given duration and sample rate"""
    key, subkey = jax.random.split(rng)
    noise_real = jax.random.normal(key, shape=(n_sample,)) * jnp.sqrt(var / 2)
    noise_imag = jax.random.normal(subkey, shape=(n_sample,)) * jnp.sqrt(var / 2)
    return jnp.fft.irfft(noise_real + 1j * noise_imag)

N_example = 7*24*60*60//4
batch_size = 10000
noise = np.concatenate(
    [generate_noise(jax.random.split(jax.random.PRNGKey(0), batch_size)) for _ in range(N_example // batch_size)]
)
noise = generate_noise(jax.random.split(jax.random.PRNGKey(0), N_example))

with h5py.File('data/noise.hdf5', 'w') as f:
    f.create_dataset('data', data=noise)
    f.attrs['duration'] = duration
    f.attrs['sampling_rate'] = sampling_rate
    f.attrs['f_min'] = f_min
    f.attrs['n_sample'] = n_sample
    f.attrs['psd'] = psd
    f.attrs['freqs'] = freqs