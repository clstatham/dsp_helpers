import numpy as np
import soundfile
import tensorflow as tf
import tensorflow.keras.backend as K

EPS = 1e-12
DEFAULT_FLOAT = tf.float32


_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 2595.0


FLOAT = lambda y: tf.cast(tf.convert_to_tensor(y), DEFAULT_FLOAT)


def write_normalized_audio_to_disk(audio, fn, sample_rate=48000):
    soundfile.write(fn, audio, sample_rate, "FLOAT", format="WAV")


@tf.function
def random_gain(audio, min_gain, max_gain):
    return audio * tf.random.uniform(
        [tf.shape(audio)[0], 1], minval=min_gain, maxval=max_gain
    )


@tf.function
def normalize_audio(audio):
    if tf.reduce_max(tf.abs(audio)) <= EPS:
        return audio
    return FLOAT(audio) / (tf.reduce_max(tf.abs(FLOAT(audio))) + EPS)


def batchwise_normalize_2d(x, return_amount=False):
    mx = tf.reduce_max(x, axis=[1, 2], keepdims=True)
    if return_amount:
        return x / max_epsilon(mx), mx
    return x / max_epsilon(mx)


@tf.function
def normalize_zero_one(x):
    a = x - tf.reduce_min(x, keepdims=True)
    b = a / (tf.reduce_max(a, keepdims=True) + 1e-9)
    return b


@tf.function
def normalize_zero_one_batchwise(x):
    out = tf.TensorArray(x.dtype, tf.shape(x)[0])
    # out = []
    for i in tf.range(tf.shape(x)[0]):
        out = out.write(i, normalize_zero_one(x[i]))
    return out.stack()


def add_epsilon(x, eps=EPS):
    return x + eps


def max_epsilon(x, eps=EPS):
    return add_epsilon(x, eps=eps)


def log_no_nan(x, eps=EPS):
    return K.log(max_epsilon(x, eps=eps))


def polar_to_rect(mag, phase_angle):
    mag = tf.complex(mag, 0.0)
    phase = tf.complex(tf.cos(phase_angle), tf.sin(phase_angle))
    return mag * phase


def mel_to_hertz(mel_values):
    """Converts frequencies in `mel_values` from the mel scale to linear scale."""
    return _MEL_BREAK_FREQUENCY_HERTZ * (
        np.exp(np.array(mel_values) / _MEL_HIGH_FREQUENCY_Q) - 1.0
    )


def hertz_to_mel(frequencies_hertz):
    """Converts frequencies in `frequencies_hertz` in Hertz to the mel scale."""
    return _MEL_HIGH_FREQUENCY_Q * np.log(
        1.0 + (np.array(frequencies_hertz) / _MEL_BREAK_FREQUENCY_HERTZ)
    )


def linear_to_mel_weight_matrix(
    num_mel_bins=128,
    num_spectrogram_bins=257,
    sample_rate=48000,
    lower_edge_hertz=_MEL_BREAK_FREQUENCY_HERTZ,
    upper_edge_hertz=_MEL_HIGH_FREQUENCY_Q,
):
    """Returns a matrix to warp linear scale spectrograms to the mel scale.
    Adapted from tf.signal.linear_to_mel_weight_matrix with a minimum
    band width (in Hz scale) of 1.5 * freq_bin. To preserve accuracy,
    we compute the matrix at float64 precision and then cast to `dtype`
    at the end. This function can be constant folded by graph optimization
    since there are no Tensor inputs.
    Args:
        num_mel_bins: Int, number of output frequency dimensions.
        num_spectrogram_bins: Int, number of input frequency dimensions.
        sample_rate: Int, sample rate of the audio.
        lower_edge_hertz: Float, lowest frequency to consider.
        upper_edge_hertz: Float, highest frequency to consider.
    Returns:
        Numpy float32 matrix of shape [num_spectrogram_bins, num_mel_bins].
    Raises:
        ValueError: Input argument in the wrong range.
    """
    # Validate input arguments
    if num_mel_bins <= 0:
        raise ValueError("num_mel_bins must be positive. Got: %s" % num_mel_bins)
    if num_spectrogram_bins <= 0:
        raise ValueError(
            "num_spectrogram_bins must be positive. Got: %s" % num_spectrogram_bins
        )
    if sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive. Got: %s" % sample_rate)
    if lower_edge_hertz < 0.0:
        raise ValueError(
            "lower_edge_hertz must be non-negative. Got: %s" % lower_edge_hertz
        )
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError(
            "lower_edge_hertz %.1f >= upper_edge_hertz %.1f"
            % (lower_edge_hertz, upper_edge_hertz)
        )
    if upper_edge_hertz > sample_rate / 2:
        raise ValueError(
            "upper_edge_hertz must not be larger than the Nyquist "
            "frequency (sample_rate / 2). Got: %s for sample_rate: %s"
            % (upper_edge_hertz, sample_rate)
        )

    # HTK excludes the spectrogram DC bin.
    bands_to_zero = 1
    nyquist_hertz = sample_rate / 2.0
    linear_frequencies = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)[
        bands_to_zero:, np.newaxis
    ]
    # spectrogram_bins_mel = hertz_to_mel(linear_frequencies)

    # Compute num_mel_bins triples of (lower_edge, center, upper_edge). The
    # center of each band is the lower and upper edge of the adjacent bands.
    # Accordingly, we divide [lower_edge_hertz, upper_edge_hertz] into
    # num_mel_bins + 2 pieces.
    band_edges_mel = np.linspace(
        hertz_to_mel(lower_edge_hertz), hertz_to_mel(upper_edge_hertz), num_mel_bins + 2
    )

    lower_edge_mel = band_edges_mel[0:-2]
    center_mel = band_edges_mel[1:-1]
    upper_edge_mel = band_edges_mel[2:]

    freq_res = nyquist_hertz / float(num_spectrogram_bins)
    freq_th = 1.5 * freq_res
    for i in range(0, num_mel_bins):
        center_hz = mel_to_hertz(center_mel[i])
        lower_hz = mel_to_hertz(lower_edge_mel[i])
        upper_hz = mel_to_hertz(upper_edge_mel[i])
        if upper_hz - lower_hz < freq_th:
            rhs = 0.5 * freq_th / (center_hz + _MEL_BREAK_FREQUENCY_HERTZ)
            dm = _MEL_HIGH_FREQUENCY_Q * np.log(rhs + np.sqrt(1.0 + rhs ** 2))
            lower_edge_mel[i] = center_mel[i] - dm
            upper_edge_mel[i] = center_mel[i] + dm

    lower_edge_hz = mel_to_hertz(lower_edge_mel)[np.newaxis, :]
    center_hz = mel_to_hertz(center_mel)[np.newaxis, :]
    upper_edge_hz = mel_to_hertz(upper_edge_mel)[np.newaxis, :]

    # Calculate lower and upper slopes for every spectrogram bin.
    # Line segments are linear in the mel domain, not Hertz.
    lower_slopes = (linear_frequencies - lower_edge_hz) / (center_hz - lower_edge_hz)
    upper_slopes = (upper_edge_hz - linear_frequencies) / (upper_edge_hz - center_hz)

    # Intersect the line segments with each other and zero.
    mel_weights_matrix = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))

    # Re-add the zeroed lower bins we sliced out above.
    # [freq, mel]
    mel_weights_matrix = np.pad(
        mel_weights_matrix, [[bands_to_zero, 0], [0, 0]], "constant"
    )
    return mel_weights_matrix


MEL_WEIGHT_MATRIX = tf.cast(
    tf.convert_to_tensor(linear_to_mel_weight_matrix()), tf.float32
)


def _mel_to_linear_matrix():
    """Get the inverse mel transformation matrix."""
    m = linear_to_mel_weight_matrix()
    m_t = np.transpose(m)
    p = np.matmul(m, m_t)
    d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
    return np.matmul(m_t, np.diag(d))


INV_MEL_WEIGHT_MATRIX = tf.cast(
    tf.convert_to_tensor(_mel_to_linear_matrix()), tf.float32
)


def l1_norm(x, axis=None):
    # return l1 * tf.reduce_sum(tf.abs(x), axis=axis)
    return tf.reduce_mean(tf.abs(x), axis=axis)


def l2_norm(x, axis=None):
    # return l2 * tf.reduce_sum(tf.square(x), axis=axis)
    return tf.math.l2_normalize(
        x, axis=axis, epsilon=EPS
    )  # tf.reduce_mean(tf.square(x), axis=axis)


def mel_to_linear(m, dim=1):
    return tf.tensordot(m, INV_MEL_WEIGHT_MATRIX, dim)


def linear_to_mel(m, dim=1):
    return tf.tensordot(m, MEL_WEIGHT_MATRIX, dim)


def amp_to_db(x):
    return 10 * tf.experimental.numpy.log10(max_epsilon(x))


def db_to_amp(x):
    return 10 ** (x / 10)


def audio_to_stft(
    x, frame_length=512, frame_step=128, window_fn=tf.signal.vorbis_window
):
    return tf.signal.stft(
        x, frame_length, frame_step, window_fn=window_fn, pad_end=False
    )


def stft_to_audio(
    x, frame_length=512, frame_step=128, window_fn=tf.signal.vorbis_window
):
    return tf.math.real(
        tf.signal.inverse_stft(x, frame_length, frame_step, window_fn=window_fn)
    )


def scaled_sigmoid_tanh_curve(x, shift=0.0, scale=1.0):
    if not tf.is_tensor(scale):
        scale = tf.convert_to_tensor(scale, dtype=tf.float32)
    x = tf.cast(x, scale.dtype)
    shift = tf.cast(shift, scale.dtype)
    return (tf.math.tanh((x + shift) / scale) + 1) / 2


def scaled_inv_sigmoid_tanh_curve(x, shift=0.0, scale=1):
    if not tf.is_tensor(scale):
        scale = tf.convert_to_tensor(scale, dtype=tf.float32)
    x = tf.cast(x, scale.dtype)
    shift = tf.cast(shift, scale.dtype)
    return tf.math.atanh(x * 2 - 1) * scale - shift


def variable_sigmoid_v1(x, shift=-1, scale=2.8):
    if not tf.is_tensor(scale):
        scale = tf.convert_to_tensor(scale, dtype=tf.float32)
    x = tf.cast(x, scale.dtype)
    shift = tf.cast(shift, scale.dtype)
    return 1 / (1 + tf.math.exp(-(x + shift) * scale))


def variable_sigmoid(x, shift=-1, scale=2.8, mode="sigmoid"):
    if mode.lower() == "sigmoid" or mode is None:
        return variable_sigmoid_v1(x, shift, scale)
    elif mode.lower() == "sigmoid2tanh":

        scale = sigmoid2tanh_scale(x, shift, scale)
        return tf.abs(scaled_sigmoid_tanh_curve(x, shift, scale))
    elif mode.lower() == "tanh":
        return scaled_sigmoid_tanh_curve(x, shift, scale)
    elif mode.lower() == "tanh2sigmoid":
        scale = tanh2sigmoid_scale(x, shift, scale)
        return tf.abs(variable_sigmoid_v1(x, shift, scale))
    else:
        raise ValueError()


# https://www.wolframalpha.com/input/?i=%28tanh%28%28z%2Ba%29%2Fb%29+%2B+1%29+%2F+2+%3D+1+%2F+%281+%2B+e%5E%28-%28z%2Ba%29+*+c%29%29+solve+for+b


def sigmoid2tanh_scale(x, shift, scale):
    return tf.cast((shift + x), tf.complex64) / (
        tf.cast(
            tf.math.atanh(
                (tf.math.exp(scale * (shift + x)) - 1)
                / (tf.math.exp(scale * (shift + x)) + 1)
            ),
            tf.complex64,
        )
        + 1j * np.pi
    )


# https://www.wolframalpha.com/input/?i=%28tanh%28%28z%2Ba%29%2Fb%29+%2B+1%29+%2F+2+%3D+1+%2F+%281+%2B+e%5E%28-%28z%2Ba%29+*+c%29%29+solve+for+c


def tanh2sigmoid_scale(x, shift, scale):
    return (
        tf.cast(
            log_no_nan(
                -(tf.math.tanh((shift + x) / scale) + 1)
                / (tf.math.tanh((shift + x) / scale) - 1)
            ),
            tf.complex64,
        )
        + 2j * np.pi
    ) / tf.cast((shift + x), tf.complex64)


def stft_to_stftgram(stft):
    mag = tf.abs(stft)
    mag = mag_to_scaled_mag(mag)
    phase_angle = tf.experimental.numpy.angle(stft)
    phase = instantaneous_frequency(phase_angle, use_unwrap=False, time_axis=-2)
    stack = tf.stack([mag, phase], -1)
    stack = tf.where(tf.math.is_nan(stack), tf.zeros_like(stack), stack)
    return stack


# @tf.function
def audio_to_stftgram(audio, frame_length=512, frame_step=128):
    return stft_to_stftgram(audio_to_stft(audio, frame_length, frame_step))


# @tf.function
def stftgram_to_stft(stftgram):
    mag = stftgram[..., 0]
    phase_angle = stftgram[..., 1]
    mag = scaled_mag_to_mag(mag)
    phase = tf.cumsum(phase_angle * np.pi, -2)
    return polar_to_rect(mag, phase)


# @tf.function
def mag_to_scaled_mag(mag, scale=64.0):
    mag = log_no_nan(mag)
    mag = scaled_sigmoid_tanh_curve(mag, shift=0.0, scale=scale)
    return mag


# @tf.function
def scaled_mag_to_mag(mag, scale=64.0):
    mag = scaled_inv_sigmoid_tanh_curve(mag, shift=0, scale=scale)
    mag = tf.math.exp(mag)
    return mag


# based on librosa's griffin-lim algorithm:
# https://librosa.org/doc/main/_modules/librosa/core/spectrum.html#griffinlim
@tf.function
def griffin_lim(stft, iters=32, momentum=0.99):
    angles = tf.exp(
        2j * np.pi * tf.cast(tf.random.normal(tf.shape(stft)), tf.complex64)
    )
    # angles = tf.exp(2j * np.pi * tf.cast(tf.math.imag(stft), tf.complex64))
    # angles = tf.complex(tf.zeros_like(tf.math.imag(stft)), tf.math.imag(stft))
    # angles = tf.experimental.numpy.angle(stft)
    rebuilt = tf.cast(tf.zeros_like(stft), tf.complex64)
    i = 0
    while i < iters:
        tprev = rebuilt
        inverse = stft_to_audio(stft * angles)
        rebuilt = audio_to_stft(inverse)
        angles = rebuilt - tf.cast((momentum / (1 + momentum)), tf.complex64) * tf.cast(
            tprev, tf.complex64
        )
        angles /= tf.cast(tf.abs(angles) + 1e-16, tf.complex64)
        i += 1

    return tf.cast(tf.math.real(stft), tf.complex64) * angles


# based on https://github.com/tuelwer/phase-retrieval/blob/master/phase_retrieval.py
def fienup(stft, iters=200, beta=0.8):
    mag = tf.cast(tf.abs(stft), tf.complex64)
    # y_hat = mag * tf.exp(1j*tf.cast(2*np.pi*tf.random.normal(mag.shape), tf.complex64))
    y_hat = stft
    x = tf.zeros_like(stft_to_audio(stft), dtype=tf.float32)
    x_p = None
    for _ in range(iters):
        y = stft_to_audio(y_hat)
        if x_p is None:
            x_p = y
        else:
            x_p = x
        x = tf.where(y < 0, x_p - beta * y, x)
        x_hat = audio_to_stft(x)
        y_hat = mag * tf.exp(
            1j * tf.cast(tf.experimental.numpy.angle(x_hat), tf.complex64)
        )
    return y_hat


# @tf.function
def stftgram_to_audio(
    stftgram,
    frame_length=512,
    frame_step=128,
):
    return stft_to_audio(
        stftgram_to_stft(stftgram), frame_length=frame_length, frame_step=frame_step
    )


# loosely based on https://github.com/timsainb/noisereduce/blob/7013c987aaf857b7e280bca8fe4fe42b112053db/noisereduce/noisereduce.py#L281
# @tf.function
def spectral_gating_nonstationary(
    stft,
    time_constant_t=0.001,
    sample_rate=48000,
    frame_step=128,
    sigmoid_shift=-1.0,
    sigmoid_scale=2.8,
    sigmoid_mode="tanh",
):
    mag = tf.abs(stft)
    t_frames = time_constant_t * float(sample_rate) / float(frame_step)
    b = (tf.sqrt(1 + 4 * t_frames ** 2) - 1) / (2 * t_frames ** 2)
    a = 1 - b
    mag_smooth = iir_filtfilt(
        mag,
        create_iir_filter_fn(
            a,
            0.0,
            0.0,
            b,
            0.0,
        ),
        poles=1,
    )
    mag_mask = mag - mag_smooth
    mag_mask = variable_sigmoid(mag_mask, sigmoid_shift, sigmoid_scale, sigmoid_mode)
    stft_denoised = stft * tf.cast(mag_mask, stft.dtype)
    return stft_denoised


# @tf.function
def spectral_gating(x_stft: tf.Tensor, mask_stft: tf.Tensor):
    out_real = tf.abs(x_stft) - tf.abs(mask_stft)
    out_real = tf.nn.relu(out_real)
    # out_real = tf.where(out_real > 0, out_real, tf.zeros_like(out_real))
    out_real *= tf.math.real(tf.sign(x_stft))
    out_imag = tf.math.imag(x_stft) - tf.math.imag(mask_stft)
    out_imag = tf.where(out_real == 0, tf.zeros_like(out_imag), out_imag)
    out_stft = tf.complex(out_real, out_imag)
    return out_stft


def convert_mag_to_sigmoid_map(x_stft, mask_mag, shift, scale, mode="tanh"):
    # sigmoid mode is required for nonstationary spectral gating;
    # however, tanh mode is more numerically stable
    mask = tf.abs(x_stft) - tf.abs(mask_mag)
    return variable_sigmoid(mask, shift, scale, mode=mode)


def stft_mask_convolution(x_stft, mask_sigmoid):
    return x_stft * tf.cast(tf.abs(mask_sigmoid), x_stft.dtype)


def spectral_gating_sigmoid(x_stft, mask_mag, shift, scale, mode="tanh"):
    mask_sigmoid = convert_mag_to_sigmoid_map(x_stft, mask_mag, shift, scale, mode=mode)
    return stft_mask_convolution(x_stft, mask_sigmoid)


@tf.function
def spectral_gating_stft(x_stft, mask_mag):
    x_phs = instantaneous_frequency(
        tf.experimental.numpy.angle(x_stft), use_unwrap=False, time_axis=-2
    )
    x_mag = tf.abs(x_stft)
    mask_mag = tf.abs(mask_mag)
    _out_mag = x_mag - mask_mag
    out_mag = tf.where(_out_mag > 0, _out_mag, tf.zeros_like(_out_mag))
    out_phs = tf.where(_out_mag > 0, x_phs, tf.zeros_like(x_phs))
    out_angular_phs = tf.cumsum(out_phs * np.pi, axis=-2)

    # if strength > 0 and boost_highs > 0:
    #     # boost high frequences based on strength and boost_highs
    #     out_mag = boost_high_freqs(out_mag, strength * boost_highs)

    out_stft = polar_to_rect(out_mag, out_angular_phs)
    return out_stft


# https://github.com/tensorflow/addons/blob/45928da883eef6e512f99e6820c56c0b2d09ee3f/tensorflow_addons/image/filters.py#L203
def get_gaussian_kernel(sigma, filter_shape):
    """Compute 1D Gaussian kernel."""
    sigma = tf.convert_to_tensor(sigma)
    x = tf.range(-filter_shape // 2 + 1, filter_shape // 2 + 1)
    x = tf.cast(x ** 2, sigma.dtype)
    x = tf.nn.softmax(-x / (2.0 * (sigma ** 2)))
    return x


def get_gaussian_kernel_2d(gaussian_filter_x, gaussian_filter_y):
    """Compute 2D Gaussian kernel given 1D kernels."""
    gaussian_kernel = tf.matmul(gaussian_filter_x, gaussian_filter_y)
    return gaussian_kernel


def convolutional_lpf(
    spectral,
    kernel_size,
    sigma=2.0,
    freq_smooth_hz=500,
    time_smooth_ms=50,
    sample_rate=48000,
    frame_length=512,
    frame_step=128,
):
    if freq_smooth_hz > 0:
        n_grad_freq = int(freq_smooth_hz / (sample_rate / (frame_length / 2)))
    else:
        n_grad_freq = 1

    n_grad_time = int(time_smooth_ms / ((frame_step / sample_rate) * 1000))
    kernel_x = get_gaussian_kernel(sigma, n_grad_freq * kernel_size)
    kernel_x = tf.expand_dims(kernel_x, 0)
    kernel_y = get_gaussian_kernel(sigma, n_grad_time * kernel_size)
    kernel_y = tf.expand_dims(kernel_y, -1)
    kernel = get_gaussian_kernel_2d(kernel_y, kernel_x)
    pad_top = (n_grad_time - 1) // 2
    pad_bottom = n_grad_time - 1 - pad_top
    pad_left = (n_grad_freq - 1) // 2
    pad_right = n_grad_freq - 1 - pad_left
    paddings = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right]]
    spectral = tf.pad(spectral, paddings, mode="REFLECT")
    kernel = tf.expand_dims(tf.expand_dims(kernel, -1), -1)
    return tf.squeeze(
        tf.nn.conv2d(
            tf.expand_dims(spectral, -1),
            kernel,
            strides=(1, 1),
            padding="VALID",
        ),
        axis=-1,
    )


@tf.function
def tensor_iir(x, a0, a1, a2, b1, b2):
    out = tf.TensorArray(x.dtype, tf.shape(x)[-1])
    val = tf.zeros_like(x[..., 0])
    val_iminus1 = tf.zeros_like(x[..., 0])
    for i in tf.range(tf.shape(x)[-1]):
        x_i = x[..., i]
        x_iminus1 = x[..., i - 1] if i > 0 else tf.zeros_like(x[..., 0])
        x_iminus2 = x[..., i - 2] if i > 1 else tf.zeros_like(x[..., 0])
        val_iminus2 = val_iminus1
        val_iminus1 = val
        val = (
            a0 * x_i
            + a1 * x_iminus1
            + a2 * x_iminus2
            - b1 * val_iminus1
            - b2 * val_iminus2
        )
        out = out.write(i, val)
    out = out.stack()
    out = tf.transpose(out, [2, 1, 0])
    return out


# @tf.function
def create_iir_filter_fn(a0, a1, a2, b1, b2):
    return lambda y: tf.transpose(
        tensor_iir(tf.transpose(y, [0, 2, 1]), a0, a1, a2, b1, b2), [0, 2, 1]
    )


def lowpass_coef(Fc, Q, sample_rate=48000):
    # V = tf.pow(10.0, tf.abs(peak_gain) / 20.0)
    K = tf.math.tan(np.pi * Fc / sample_rate)

    norm = 1.0 / (1.0 + K / Q + K * K)
    a0 = K * K * norm
    a1 = 2 * a0
    a2 = a0
    b1 = 2 * (K * K - 1) * norm
    b2 = (1 - K / Q + K * K) * norm
    return a0, a1, a2, b1, b2


def highpass_coef(Fc, Q, sample_rate=48000):
    # V = tf.pow(10.0, tf.abs(peak_gain) / 20.0)
    K = tf.math.tan(np.pi * Fc / sample_rate)
    norm = 1.0 / (1.0 + K / Q + K * K)
    a0 = 1 * norm
    a1 = -2 * a0
    a2 = a0
    b1 = 2 * (K * K - 1) * norm
    b2 = (1 - K / Q + K * K) * norm
    return a0, a1, a2, b1, b2


def create_lowpass_filter_fn(Fc, Q):
    return create_iir_filter_fn(*lowpass_coef(Fc, Q))


def create_highpass_filter_fn(Fc, Q):
    return create_iir_filter_fn(*highpass_coef(Fc, Q))


def run_iir_filter(x, filter_fn, poles=2):
    for _ in range(poles):
        x = filter_fn(x)
    return x


def iir_filtfilt(x, filter_fn, poles=2):
    return tf.reverse(
        run_iir_filter(
            tf.reverse(run_iir_filter(x, filter_fn, poles), [-1]), filter_fn, poles
        ),
        [-1],
    )


def run_fir_filter(x, iir_filter_fn, ir_len=None, poles=2):
    x = tf.transpose(x, [0, 2, 1])
    if ir_len is None:
        ir_len = tf.shape(x)[-1]
    ir = tf.constant([1.0])
    ir = tf.concat([ir, tf.zeros(ir_len - 1)], axis=0)
    ir = run_iir_filter(ir, iir_filter_fn, poles)
    ir = tf.reshape(ir, [1, 1, -1])
    x_fft = tf.signal.rfft2d(x)
    ir_fft = tf.signal.rfft2d(ir)
    ret = tf.signal.irfft2d(x_fft * ir_fft)
    ret = tf.concat([ret, ret[..., -2:-1]], axis=-1)
    ret = tf.transpose(ret, [0, 2, 1])
    return ret


def fir_filtfilt_2d(x, iir_filter_fn, ir_len=None, poles=2):
    return tf.reverse(
        run_fir_filter(
            tf.reverse(run_fir_filter(x, iir_filter_fn, ir_len, poles), [-1]),
            iir_filter_fn,
            ir_len,
            poles,
        ),
        [-1],
    )


def fir_convolution_1d(x, iir_filter_fn, ir_len, poles=2):
    ir = tf.constant([1.0])
    ir = tf.concat([ir, tf.zeros(ir_len - 1)], axis=0)
    ir = tf.expand_dims(ir, 0)
    ir = run_iir_filter(ir, iir_filter_fn, poles)
    ir = tf.reshape(ir, [-1, 1, 1])
    batch_size = tf.shape(x)[0]
    x = tf.reshape(x, [batch_size, -1, 1])
    x = tf.pad(
        x, [[0, 0], [(ir_len - 1) // 2, (ir_len - 1) // 2], [0, 0]], mode="REFLECT"
    )
    return tf.squeeze(tf.nn.conv1d(x, ir, 1, "VALID"))


def fir_convolution_2d(x, iir_filter_fn, ir_len, poles=2):
    x = tf.transpose(x, [0, 1, 2])
    ir = tf.constant([1.0])
    ir = tf.concat([ir, tf.zeros(ir_len - 1)], axis=0)
    ir = tf.expand_dims(ir, 0)
    ir = run_iir_filter(ir, iir_filter_fn, poles)
    ir = tf.reshape(ir, [1, -1, 1, 1])
    ir = tf.repeat(ir, x.shape[1], 0)
    # batch_size = tf.shape(x)[0]
    # assert batch_size <= BATCH_SIZE or batch_size is None
    # x = tf.reshape(x, [batch_size, -1, 1])
    x = tf.expand_dims(x, -1)
    # if axis == -1 or axis == 2:
    x = tf.pad(
        x,
        [[0, 0], [0, 0], [(ir_len - 1) // 2, (ir_len - 1) // 2], [0, 0]],
        mode="REFLECT",
    )
    return tf.transpose(tf.squeeze(tf.nn.conv2d(x, ir, 1, "VALID"), -1), [0, 1, 2])


def fir_filtfilt_convolution_2d(x, iir_filter_fn, ir_len, poles=2):
    x = fir_convolution_2d(x, iir_filter_fn, ir_len, poles)
    x = tf.reverse(x, [-1])
    x = fir_convolution_2d(x, iir_filter_fn, ir_len, poles)
    x = tf.reverse(x, [-1])
    return x


@tf.function
def diff(x, axis=-1):
    # TODO: add more axes
    if axis == -1:
        return x[..., 1:] - x[..., :-1]
    elif axis == -2:
        return x[..., 1:, :] - x[..., :-1, :]
    elif axis == -3:
        return x[..., 1:, :, :] - x[..., :-1, :, :]
    elif axis == 0:
        return x[1:] - x[:-1]
    elif axis == 1:
        return x[:, 1:] - x[:, :-1]
    elif axis == 2:
        return x[:, :, 1:] - x[:, :, :-1]
    elif axis == 3:
        return x[:, :, :, 1:] - x[:, :, :, :-1]
    else:
        raise ValueError()


# https://github.com/magenta/magenta/blob/c1340b2788af9bc193ef23e1ecec3fabf13d0a14/magenta/models/gansynth/lib/spectral_ops.py#L142
def unwrap(p, discont=np.pi, axis=-1):
    """Unwrap a cyclical phase tensor.
    Args:
        p: Phase tensor.
        discont: Float, size of the cyclic discontinuity.
        axis: Axis of which to unwrap.
    Returns:
        unwrapped: Unwrapped tensor of same size as input.
    """
    dd = diff(p, axis=axis)
    ddmod = tf.math.mod(dd + np.pi, 2.0 * np.pi) - np.pi
    idx = tf.logical_and(tf.equal(ddmod, -np.pi), tf.greater(dd, 0))
    ddmod = tf.where(idx, tf.ones_like(ddmod) * np.pi, ddmod)
    ph_correct = ddmod - dd
    idx = tf.less(tf.abs(dd), discont)
    ddmod = tf.where(idx, tf.zeros_like(ddmod), dd)
    ph_cumsum = tf.cumsum(ph_correct, axis=axis)

    if axis == -1:
        shape = tf.shape(p[..., 0:1])
    elif axis == -2:
        shape = tf.shape(p[..., 0:1, :])
    elif axis == -3:
        shape = tf.shape(p[..., 0:1, :, :])
    else:
        raise ValueError()

    ph_cumsum = tf.concat([tf.zeros(shape, dtype=p.dtype), ph_cumsum], axis=axis)
    return p + ph_cumsum


@tf.function
def instantaneous_frequency(phase_angle, time_axis=-2, use_unwrap=True):
    """Transform a fft tensor from phase angle to instantaneous frequency.
    Take the finite difference of the phase. Pad with initial phase to keep the
    tensor the same size.
    Args:
        phase_angle: Tensor of angles in radians. [Batch, Time, Freqs]
        time_axis: Axis over which to unwrap and take finite difference.
        use_unwrap: True preserves original GANSynth behavior, whereas False will
            guard against loss of precision.
    Returns:
        dphase: Instantaneous frequency (derivative of phase). Same size as input.
    """
    if use_unwrap:
        # Can lead to loss of precision.
        phase_unwrapped = unwrap(phase_angle, axis=time_axis)
        dphase = diff(phase_unwrapped, axis=time_axis)
    else:
        # Keep dphase bounded. N.B. runs faster than a single mod-2pi expression.
        dphase = diff(phase_angle, axis=time_axis)
        dphase = tf.where(dphase > np.pi, dphase - 2 * np.pi, dphase)
        dphase = tf.where(dphase < -np.pi, dphase + 2 * np.pi, dphase)

    # Add an initial phase to dphase.
    if time_axis == -1:
        phase_slice = phase_angle[..., 0:1]
    elif time_axis == -2:
        phase_slice = phase_angle[..., 0:1, :]
    elif time_axis == -3:
        phase_slice = phase_angle[..., 0:1, :, :]
    else:
        raise ValueError()
    dphase = tf.concat([phase_slice, dphase], axis=time_axis) / np.pi
    return dphase
