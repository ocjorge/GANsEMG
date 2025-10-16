# =============================================================================
# SCRIPT DE ENTRENAMIENTO DE LA GAN (VERSIÓN LOCAL FINAL Y ROBUSTA)
# =============================================================================

# --- SECCIÓN 1: IMPORTACIONES ---
import os
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time
import random
from tensorflow.keras.layers import Input, Conv1D, Conv1DTranspose, LeakyReLU, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# --- SECCIÓN 2: CONFIGURACIÓN ---
DATASET_PATH_DB1 = 'ninapro_db1_data'
DATASET_PATH_DB3 = 'ninapro_db3_data'
WINDOW_SIZE, NUM_CHANNELS = 200, 12
INPUT_SHAPE = (WINDOW_SIZE, NUM_CHANNELS)
EPOCHS, BATCH_SIZE = 50, 32
APPROX_STEPS_PER_EPOCH = 5000


# --- SECCIÓN 3: FUNCIONES DE CARGA Y GENERACIÓN DE DATOS ---

def load_data_for_subject(base_path, subject_id):
    """
    Función robusta unificada que carga datos encontrando archivos por patrón.
    """
    all_emg, all_gestures = np.array([]), np.array([])
    try:
        all_files_in_dir = os.listdir(base_path)
    except FileNotFoundError:
        return None, None
    subject_pattern = f'S{subject_id}_'
    subject_files = [f for f in all_files_in_dir if f.startswith(subject_pattern) and f.endswith('.mat')]
    if not subject_files:
        return None, None
    for filename in sorted(subject_files):
        file_path = os.path.join(base_path, filename)
        try:
            data = loadmat(file_path)
            if 'emg' in data and 'restimulus' in data:
                emg, gestures = data['emg'], data['restimulus']
                if all_emg.size == 0:
                    all_emg, all_gestures = emg, gestures
                else:
                    all_emg = np.vstack((all_emg, emg))
                    all_gestures = np.vstack((all_gestures, gestures))
        except Exception:
            continue
    if all_emg.size == 0:
        return None, None
    return all_emg, all_gestures


def data_generator(base_path, subjects, target_channels, is_sano_domain):
    """
    Generador de Python que produce ventanas de datos una por una, con mensajes de seguimiento.
    """
    domain_name = "Sano (DB1)" if is_sano_domain else "Amputado (DB3)"
    subject_list = list(subjects)

    while True:
        random.shuffle(subject_list)
        for subject_id in subject_list:
            print(f"    [Generador - Dominio {domain_name}] Intentando cargar datos para Sujeto {subject_id}...")

            emg_data, gesture_labels = load_data_for_subject(base_path, subject_id)

            if emg_data is None:
                print(
                    f"    [Generador - Dominio {domain_name}] No se encontraron datos para Sujeto {subject_id}. Omitiendo.")
                continue

            print(
                f"    [Generador - Dominio {domain_name}] Datos cargados para Sujeto {subject_id}. Empezando a generar ventanas.")

            active_indices = np.where(gesture_labels.flatten() != 0)[0]
            indices_list = list(range(0, len(active_indices) - WINDOW_SIZE, 50))
            random.shuffle(indices_list)

            for i in indices_list:
                window_indices = active_indices[i: i + WINDOW_SIZE]
                if window_indices[-1] - window_indices[0] != WINDOW_SIZE - 1: continue

                window_emg = emg_data[window_indices]

                if window_emg.shape[1] < target_channels:
                    padding = np.zeros((window_emg.shape[0], target_channels - window_emg.shape[1]))
                    window_emg = np.concatenate([window_emg, padding], axis=1)

                min_val, max_val = np.min(window_emg), np.max(window_emg)
                window_normalized = 2 * (window_emg - min_val) / (max_val - min_val + 1e-8) - 1

                yield window_normalized


# --- SECCIÓN 4: CONSTRUCCIÓN DE MODELOS ---
def build_generator():
    inputs = Input(shape=INPUT_SHAPE)
    e1 = Conv1D(filters=32, kernel_size=5, strides=2, padding='same')(inputs);
    e1 = LeakyReLU(alpha=0.2)(e1)
    e2 = Conv1D(filters=64, kernel_size=5, strides=2, padding='same')(e1);
    e2 = BatchNormalization()(e2);
    e2 = LeakyReLU(alpha=0.2)(e2)
    e3 = Conv1D(filters=128, kernel_size=5, strides=2, padding='same')(e2);
    e3 = BatchNormalization()(e3);
    e3 = LeakyReLU(alpha=0.2)(e3)
    d1 = Conv1DTranspose(filters=64, kernel_size=5, strides=2, padding='same')(e3);
    d1 = BatchNormalization()(d1);
    d1 = Dropout(0.2)(d1);
    d1 = Concatenate()([d1, e2]);
    d1 = LeakyReLU(alpha=0.2)(d1)
    d2 = Conv1DTranspose(filters=32, kernel_size=5, strides=2, padding='same')(d1);
    d2 = BatchNormalization()(d2);
    d2 = Dropout(0.2)(d2);
    d2 = Concatenate()([d2, e1]);
    d2 = LeakyReLU(alpha=0.2)(d2)
    output = Conv1DTranspose(filters=NUM_CHANNELS, kernel_size=5, strides=2, padding='same', activation='tanh')(d2)
    return Model(inputs, output)


def build_discriminator():
    inputs = Input(shape=INPUT_SHAPE)
    d = Conv1D(filters=64, kernel_size=5, strides=2, padding='same')(inputs);
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv1D(filters=128, kernel_size=5, strides=2, padding='same')(d);
    d = BatchNormalization()(d);
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv1D(filters=256, kernel_size=5, strides=2, padding='same')(d);
    d = BatchNormalization()(d);
    d = LeakyReLU(alpha=0.2)(d)
    output = Conv1D(filters=1, kernel_size=5, padding='same')(d)
    return Model(inputs, output)


# --- SECCIÓN 5: LÓGICA DE ENTRENAMIENTO ---
if __name__ == "__main__":
    print("Configurando los generadores de datos...")
    output_signature = tf.TensorSpec(shape=(WINDOW_SIZE, NUM_CHANNELS), dtype=tf.float32)

    train_sanos = tf.data.Dataset.from_generator(
        lambda: data_generator(DATASET_PATH_DB1, range(1, 28), NUM_CHANNELS, True),
        output_signature=output_signature
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    train_amputados = tf.data.Dataset.from_generator(
        lambda: data_generator(DATASET_PATH_DB3, range(1, 12), NUM_CHANNELS, False),
        output_signature=output_signature
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    print("Datasets listos para el entrenamiento.")

    g_AB = build_generator()  # Sano -> Amputado
    g_BA = build_generator()  # Amputado -> Sano (el que nos interesa)
    d_A = build_discriminator()
    d_B = build_discriminator()

    loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    mae = tf.keras.losses.MeanAbsoluteError()
    g_optimizer_AB = Adam(2e-4, beta_1=0.5)
    g_optimizer_BA = Adam(2e-4, beta_1=0.5)
    d_optimizer_A = Adam(2e-4, beta_1=0.5)
    d_optimizer_B = Adam(2e-4, beta_1=0.5)


    def discriminator_loss(real, generated):
        return loss_obj(tf.ones_like(real), real) + loss_obj(tf.zeros_like(generated), generated)


    def generator_loss(generated):
        return loss_obj(tf.ones_like(generated), generated)


    def cycle_loss(real, cycled):
        return mae(real, cycled)


    def identity_loss(real, same):
        return mae(real, same)


    @tf.function
    def train_step(real_sano, real_amputado):
        with tf.GradientTape(persistent=True) as tape:
            fake_amputado = g_AB(real_sano, training=True)
            cycled_sano = g_BA(fake_amputado, training=True)
            fake_sano = g_BA(real_amputado, training=True)
            cycled_amputado = g_AB(fake_sano, training=True)
            same_sano = g_AB(real_sano, training=True)
            same_amputado = g_BA(real_amputado, training=True)
            disc_real_sano = d_A(real_sano, training=True)
            disc_fake_sano = d_A(fake_sano, training=True)
            disc_real_amputado = d_B(real_amputado, training=True)
            disc_fake_amputado = d_B(fake_amputado, training=True)
            gen_AB_loss = generator_loss(disc_fake_amputado)
            gen_BA_loss = generator_loss(disc_fake_sano)
            total_cycle_loss = (cycle_loss(real_sano, cycled_sano) * 10.0) + (
                        cycle_loss(real_amputado, cycled_amputado) * 10.0)
            total_identity_loss = (identity_loss(real_sano, same_sano) * 5.0) + (
                        identity_loss(real_amputado, same_amputado) * 5.0)
            total_gen_AB_loss = gen_AB_loss + total_cycle_loss + total_identity_loss
            total_gen_BA_loss = gen_BA_loss + total_cycle_loss + total_identity_loss
            disc_A_loss = discriminator_loss(disc_real_sano, disc_fake_sano)
            disc_B_loss = discriminator_loss(disc_real_amputado, disc_fake_amputado)
        g_grads_AB = tape.gradient(total_gen_AB_loss, g_AB.trainable_variables)
        g_grads_BA = tape.gradient(total_gen_BA_loss, g_BA.trainable_variables)
        d_grads_A = tape.gradient(disc_A_loss, d_A.trainable_variables)
        d_grads_B = tape.gradient(disc_B_loss, d_B.trainable_variables)
        g_optimizer_AB.apply_gradients(zip(g_grads_AB, g_AB.trainable_variables))
        g_optimizer_BA.apply_gradients(zip(g_grads_BA, g_BA.trainable_variables))
        d_optimizer_A.apply_gradients(zip(d_grads_A, d_A.trainable_variables))
        d_optimizer_B.apply_gradients(zip(d_grads_B, d_B.trainable_variables))
        return gen_BA_loss, disc_A_loss, total_cycle_loss


    print("\nIniciando entrenamiento de la GAN en local...")
    for epoch in range(EPOCHS):
        start = time.time()
        n = 0
        for batch_sano, batch_amputado in zip(train_sanos, train_amputados):
            losses = train_step(batch_sano, batch_amputado)
            if n % 200 == 0:
                print(
                    f"  Epoch {epoch + 1}, Step {n}: Gen_loss: {losses[0]:.4f}, Disc_loss: {losses[1]:.4f}, Cycle_loss: {losses[2]:.4f}")
            n += 1
            if n > APPROX_STEPS_PER_EPOCH:
                break
        print(f"Time for epoch {epoch + 1} is {time.time() - start:.2f} sec")
        if (epoch + 1) % 5 == 0:
            g_BA.save(f'generador_gan_epoch_{epoch + 1}.h5')

    g_BA.save('generador_gan.h5')
    print("Entrenamiento de la GAN completado. Modelo 'generador_gan.h5' guardado.")

    print("\nGenerando un ejemplo de reconstrucción...")
    sample_amputado_gen = data_generator(DATASET_PATH_DB3, [11], NUM_CHANNELS, False)
    sample_amputado = next(sample_amputado_gen)
    sample_amputado = np.expand_dims(sample_amputado, axis=0)
    reconstructed_sano = g_BA.predict(sample_amputado)

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.title("Señal Original de Amputado (Canal 0)")
    plt.plot(sample_amputado[0, :, 0])
    plt.subplot(1, 2, 2)
    plt.title("Señal Reconstruida como 'Sana' (Canal 0)")
    plt.plot(reconstructed_sano[0, :, 0])
    plt.show()

    
