import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import sys
import csv
from datetime import datetime
import os
import torch
from denoiser.dsp import convert_audio

sys.path.append("../../utils")
import noise
from metrics import SNR, signal_power

from tools import (
    find_oracle_threshold,
    denoise_spectral_sub,
    demucs_denoise,
    deepinv_sure_param_search,
    SpectralSubNumpyWrapper,
    deepinv_sure_spectral_sub_threshold_search,
    deepinv_sure_demucs_drywet_search_cached,
    deepinv_sure_spectral_sub_threshold_search_torch,
    demucs_sure_denoise,
    sgmse_sure_denoise,
    estimate_sigma_unsure_audio,
    robust_sigma_init_from_differences,
)

epath = "../../../results/" + datetime.now().strftime("%Y%m%d_%H%M") + "_experience_oSNR_sigma_sure_soustraction_spectrale_demucs"
os.mkdir(epath)

CSV_export = []

parameters = open('params_multiple.txt')
params = {}
for full_line in parameters:
    line = full_line.strip()
    if not line[0] == '#':
        key, value = line.split('=', 1)
        val = value.strip().split(',')
        if len(val) != 1:
            for i in range(len(val)):
                val[i] = float(val[i].strip())
        else:
            val = val[0].strip()
            if val.isdigit():
                val = int(val)
            elif val == 'True':
                val = True
            elif val == 'False':
                val = False
            else:
                try:
                    val = float(val)
                except:
                    val = val
        params[key.strip()] = val

if type(params['isnr']) != list:
    params['isnr'] = [params['isnr']]


def clustered_logspace(N, low=1e-3, high=1.0, band=(5e-2, 2e-1), frac_in_band=0.65):
    if N < 3:
        return np.logspace(np.log10(low), np.log10(high), N)

    b0, b1 = band

    n_mid = int(round(frac_in_band * N))
    n_side = N - n_mid
    n_low = n_side // 2
    n_high = n_side - n_low

    parts = []

    if n_low > 0:
        parts.append(np.logspace(np.log10(low), np.log10(b0), n_low, endpoint=False))

    n_mid_left = n_mid // 2
    n_mid_right = n_mid - n_mid_left

    parts.append(np.logspace(np.log10(b0), np.log10(1e-1), n_mid_left, endpoint=False))
    parts.append(np.logspace(np.log10(1e-1), np.log10(b1), n_mid_right, endpoint=False))

    if n_high > 0:
        parts.append(np.logspace(np.log10(b1), np.log10(high), n_high))

    return np.concatenate(parts)

def clustered_logspace(
    N,
    start=1e-3,
    stop=1.0,
    #band=(3e-2, 3e-1),#isnr 5
    band=(5e-2, 4e-1),#isnr 0
    #band=(1e-2, 1e-1),#isnr 10
    band_fraction=0.75,
    left_tail_fraction=0.4
):
    """
    Door-shaped logspace distribution.

    Most points are spread evenly in log-space across `band`.
    The remaining points are split between the left and right tails.

    left_tail_fraction controls how many outside-band points go left.

    left_tail_fraction = 0.5  -> equal tails
    left_tail_fraction = 0.35 -> fewer left-tail points, more right-tail points
    left_tail_fraction = 0.2  -> much fewer left-tail points
    """

    if N < 1:
        raise ValueError("N must be at least 1")

    band_low, band_high = band

    if not (start > 0 and stop > 0 and band_low > 0 and band_high > 0):
        raise ValueError("All values must be positive")

    if not (start < band_low < band_high < stop):
        raise ValueError("Require start < band_low < band_high < stop")

    if not (0 <= band_fraction <= 1):
        raise ValueError("band_fraction must be between 0 and 1")

    if not (0 <= left_tail_fraction <= 1):
        raise ValueError("left_tail_fraction must be between 0 and 1")

    if N == 1:
        return np.array([np.sqrt(band_low * band_high)])

    n_band = int(round(N * band_fraction))
    n_band = max(2, min(N, n_band))

    n_tail = N - n_band

    n_left = int(round(n_tail * left_tail_fraction))
    n_right = n_tail - n_left

    parts = []

    if n_left > 0:
        left = np.logspace(
            np.log10(start),
            np.log10(band_low),
            n_left,
            endpoint=False
        )
        parts.append(left)

    band_points = np.logspace(
        np.log10(band_low),
        np.log10(band_high),
        n_band,
        endpoint=True
    )
    parts.append(band_points)

    if n_right > 0:
        right = np.logspace(
            np.log10(band_high),
            np.log10(stop),
            n_right + 1,
            endpoint=True
        )[1:]
        parts.append(right)

    return np.concatenate(parts)

def select_sgmse_clip(
    sound,
    noisy_sound,
    samplerate,
    duration=1.5,
    min_power_fraction=0.3,
    max_rerolls=30,
):
    sgmse_len = int(duration * samplerate)

    if len(sound) <= sgmse_len:
        sound_clip = sound[:sgmse_len]
        noisy_clip = noisy_sound[:sgmse_len]
        clip_power = signal_power(sound_clip)
        return noisy_clip, sound_clip, 0, clip_power, clip_power, True

    full_power = signal_power(sound)
    min_power = min_power_fraction * full_power

    max_start = len(sound) - sgmse_len

    best_start = 0
    best_power = -np.inf

    for _ in range(max_rerolls):
        start = np.random.randint(0, max_start + 1)

        sound_clip = sound[start:start + sgmse_len]
        clip_power = signal_power(sound_clip)

        if clip_power > best_power:
            best_power = clip_power
            best_start = start

        if clip_power >= min_power:
            noisy_clip = noisy_sound[start:start + sgmse_len]
            return noisy_clip, sound_clip, start, clip_power, min_power, True

    sound_clip = sound[best_start:best_start + sgmse_len]
    noisy_clip = noisy_sound[best_start:best_start + sgmse_len]

    return noisy_clip, sound_clip, best_start, best_power, min_power, False

params["sgmse_duration"] = 1.5
params["sgmse_min_power_fraction"] = 0.1
params["sgmse_clip_max_rerolls"] = 30
    

params["sigma_specdemucs"] = clustered_logspace(2)
params["sigma_sgmse"] = clustered_logspace(14)
params['drywet'] = np.linspace(0, 1, 100)

params["unsure_sigma_steps"] = 200
params["unsure_sigma_batch_size"] = 32
params["unsure_sigma_patch_size"] = 8192
params["unsure_sigma_lr"] = 1e-3
params["unsure_sigma_tau"] = 1e-3
params["unsure_sigma_step_size"] = 1e-4
params["unsure_sigma_momentum"] = 0.9

params["sgmse_checkpoint"] = "../../../data/sgmse_voicebank.ckpt"
params["sgmse_sure_steps"] = 50
params["sgmse_sure_lr"] = 8e-5
params["sgmse_sure_N"] = 1
params["sgmse_sure_mc_batch_size"] = 1
params["sgmse_sure_tau"] = 1e-3
params["sgmse_snr"] = 0.5
params["sgmse_corrector"] = "ald"
params["sgmse_corrector_steps"] = 1
params["sgmse_t_eps"] = 0.03

sound_folder = '../../../data/' + params['pure_sound_folder']
sound_files = os.listdir(sound_folder)

sound_files_clean = []
for folder in sorted(sound_files):
    wav_path = os.path.join(sound_folder, folder, params["sound_type"] + ".wav")
    if os.path.isfile(wav_path):
        sound_files_clean.append(os.path.join(folder, params["sound_type"] + ".wav"))
    if len(sound_files_clean) >= params["track_samples"]:
        break

all_oSNR = {}
all_oSNR_demucs = {}
all_oSNR_sgmse = {}
all_sure_thresholds = {}
all_sigma_true = {}
all_sigma_estimate = {}

for isnr in params['isnr']:
    sound_number = 0
    all_oSNR[isnr] = []
    all_oSNR_demucs[isnr] = []
    all_oSNR_sgmse[isnr] = []
    all_sure_thresholds[isnr] = []
    all_sigma_true[isnr] = []
    all_sigma_estimate[isnr] = []

    for sound_file in sound_files_clean:
        sound_number += 1

        info = sf.info(sound_folder + '/' + sound_file)
        samplerate = info.samplerate

        duration_extract = 10
        extract_length = duration_extract * samplerate

        if info.frames > extract_length:
            start = np.random.randint(0, info.frames - extract_length)
            sound, samplerate = sf.read(sound_folder + '/' + sound_file, start=start, frames=extract_length)
        else:
            sound, samplerate = sf.read(sound_folder + '/' + sound_file)

        if len(sound.shape) > 1:
            sound = np.mean(sound, axis=1)

        sound = convert_audio(torch.from_numpy(sound).float()[None, :], samplerate, 16000, 1)[0].detach().cpu().numpy()
        samplerate = 16000

        noisy_sound, sigma_true = noise.add_white_noise(sound, samplerate, isnr)

        if sigma_true < 10**-2:
            continue

        sigma_estimate = estimate_sigma_unsure_audio(
            noisy_sound,
            sigma_init=robust_sigma_init_from_differences(noisy_sound),
            steps=params["unsure_sigma_steps"],
            batch_size=params["unsure_sigma_batch_size"],
            patch_size=params["unsure_sigma_patch_size"],
            lr=params["unsure_sigma_lr"],
            tau=params["unsure_sigma_tau"],
            unsure_step_size=params["unsure_sigma_step_size"],
            unsure_momentum=params["unsure_sigma_momentum"],
            verbose=False,
        )

        print(
            f"Sigma | extract {sound_number}/{len(sound_files_clean)} | "
            f"iSNR={isnr} | true={sigma_true:.4e} | UNSURE={sigma_estimate:.4e}",
            flush=True,
        )

        oracle_threshold, figure, thresholds, oSNR_threshold_list = find_oracle_threshold(
            sound,
            noisy_sound,
            sigma_true,
            samplerate,
            denoise_spectral_sub,
            True
        )
        plt.close(figure)

        oSNR_sigma_list = []
        oSNR_sigma_list_demucs = []
        sure_threshold_list = []
        oSNR_sigma_list_sgmse = []

        for sigma in params["sigma_specdemucs"]:

            denoised_sure, sure_threshold, sure_values = deepinv_sure_spectral_sub_threshold_search_torch(
                y_np=noisy_sound,
                thresholds=thresholds,
                sigma=sigma,
                fs=samplerate,
                chunk_size=65536,
                tau=0.01,
                n_sure_repeats=1,
            )

            _, denoised_sure_specsub = denoise_spectral_sub(
                noisy_sound,
                sigma,
                sure_threshold,
                samplerate
            )

            osnr = SNR(sound, denoised_sure_specsub)

            oSNR_sigma_list.append(osnr)
            sure_threshold_list.append(sure_threshold)

            denoised_sure_demucs = demucs_sure_denoise(
                noisy_sound,
                sigma,
                steps=7,
                lr=3e-4,
                tau=1e-3,
                mc_batch_size=5,
                debug=False,
            )

            osnr_demucs = SNR(sound, denoised_sure_demucs)

            oSNR_sigma_list_demucs.append(osnr_demucs)

            del denoised_sure_demucs

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        noisy_sound_sgmse, sound_sgmse, sgmse_start, sgmse_clip_power, sgmse_min_power, sgmse_power_ok = select_sgmse_clip(
            sound,
            noisy_sound,
            samplerate,
            duration=params["sgmse_duration"],
            min_power_fraction=params["sgmse_min_power_fraction"],
            max_rerolls=params["sgmse_clip_max_rerolls"],
        )
        
        print(
            f"SGMSE clip | extract {sound_number}/{len(sound_files_clean)} | "
            f"start={sgmse_start / samplerate:.3f}s | "
            f"power={sgmse_clip_power:.4e} | min={sgmse_min_power:.4e} | "
            f"accepted={sgmse_power_ok}",
            flush=True,
        )
        
        for sigma in params["sigma_sgmse"]:

            print(
                f"SGMSE SURE | extract {sound_number}/{len(sound_files_clean)} | "
                f"iSNR={isnr} | sigma={sigma:.3e}",
                flush=True,
            )
            '''
            sgmse_duration = 1.5
            sgmse_len = int(sgmse_duration * samplerate)

            noisy_sound_sgmse = noisy_sound[:sgmse_len]
            sound_sgmse = sound[:sgmse_len]
            '''
            denoised_sure_sgmse = sgmse_sure_denoise(
                audio_np=noisy_sound_sgmse,
                sigma=sigma,
                checkpoint_path=params["sgmse_checkpoint"],
                fs=samplerate,
                steps=params["sgmse_sure_steps"],
                lr=params["sgmse_sure_lr"],
                tau=params["sgmse_sure_tau"],
                mc_batch_size=params["sgmse_sure_mc_batch_size"],
                sgmse_N=params["sgmse_sure_N"],
                corrector=params["sgmse_corrector"],
                corrector_steps=params["sgmse_corrector_steps"],
                snr=params["sgmse_snr"],
                t_eps=params["sgmse_t_eps"],
                debug=False,
            )

            denoised_sure_sgmse = denoised_sure_sgmse[:len(sound_sgmse)]
            #if 3e-2<sigma <= 2e-1:
            if sigma <= 2e-1:
                drywets = np.linspace(0, 1, 10)

                best_drywet_sgmse = 0
                best_osnr_sgmse = -np.inf

                for drywet_sgmse in drywets:
                    denoised_sgmse_drywet = (
                        drywet_sgmse * denoised_sure_sgmse
                        + (1.0 - drywet_sgmse) * noisy_sound_sgmse
                    )

                    osnr_sgmse_test = SNR(
                        sound_sgmse,
                        denoised_sgmse_drywet,
                    )

                    if np.isfinite(osnr_sgmse_test) and osnr_sgmse_test > best_osnr_sgmse:
                        best_osnr_sgmse = osnr_sgmse_test
                        best_drywet_sgmse = drywet_sgmse

                if not np.isfinite(best_osnr_sgmse):
                    print("WARNING: SGMSE oSNR is NaN/inf, replacing with noisy baseline")
                    best_drywet_sgmse = 0
                    best_osnr_sgmse = SNR(sound_sgmse, noisy_sound_sgmse)

                osnr_sgmse = best_osnr_sgmse

                print(
                    f"SGMSE best drywet={best_drywet_sgmse:.3f} | "
                    f"oSNR={osnr_sgmse:.3f} dB",
                    flush=True,
                )

            else:
                osnr_sgmse = SNR(
                    sound_sgmse,
                    denoised_sure_sgmse,
                )

                if not np.isfinite(osnr_sgmse):
                    print("WARNING: SGMSE oSNR is NaN/inf, replacing with 0 dB")
                    osnr_sgmse = 0

                osnr_sgmse = max(osnr_sgmse, 0)

                print(
                    f"SGMSE no drywet | "
                    f"oSNR={osnr_sgmse:.3f} dB",
                    flush=True,
                )

            del denoised_sure_sgmse

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            oSNR_sigma_list_sgmse.append(osnr_sgmse)

        sigma_array_specdemucs = np.array(params["sigma_specdemucs"])
        sigma_array_sgmse = np.array(params["sigma_sgmse"])

        oSNR_sigma_array = np.array(oSNR_sigma_list)
        oSNR_sigma_array_demucs = np.array(oSNR_sigma_list_demucs)
        oSNR_sigma_array_sgmse = np.array(oSNR_sigma_list_sgmse)

        mask_specdemucs = sigma_array_specdemucs > 0
        mask_sgmse = sigma_array_sgmse > 0

        figure = plt.figure(figsize=(8, 5))

        plt.semilogx(
            sigma_array_specdemucs[mask_specdemucs],
            oSNR_sigma_array[mask_specdemucs],
            label='Soustraction spectrale, iSNR = ' + str(isnr) + ' dB'
        )

        plt.semilogx(
            sigma_array_specdemucs[mask_specdemucs],
            oSNR_sigma_array_demucs[mask_specdemucs],
            label='DEMUCS, iSNR = ' + str(isnr) + ' dB'
        )

        plt.semilogx(
            sigma_array_sgmse[mask_sgmse],
            oSNR_sigma_array_sgmse[mask_sgmse],
            label='SGMSE, iSNR = ' + str(isnr) + ' dB'
        )

        plt.axvline(
            sigma_true,
            color='black',
            linestyle='--',
            linewidth=2,
            label='True sigma'
        )

        plt.axvline(
            sigma_estimate,
            color='red',
            linestyle='--',
            linewidth=2,
            label='UNSURE estimated sigma'
        )

        plt.xlabel('Sigma given to SURE')
        plt.ylabel('oSNR')
        plt.title('oSNR as function of sigma given to SURE - ' + str(sound_number))
        plt.grid(True, which='both')
        plt.legend()

        figure.savefig(
            epath + "/individual_oSNR_sigma_" + str(sound_number) + "_iSNR_" + str(isnr) + ".jpg",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(figure)

        all_oSNR[isnr].append(oSNR_sigma_list)
        all_oSNR_demucs[isnr].append(oSNR_sigma_list_demucs)
        all_oSNR_sgmse[isnr].append(oSNR_sigma_list_sgmse)
        all_sure_thresholds[isnr].append(sure_threshold_list)
        all_sigma_true[isnr].append(sigma_true)
        all_sigma_estimate[isnr].append(sigma_estimate)

plt.figure(figsize=(8, 5))

CSV_export.append([
    'iSNR',
    'Sigma given to SURE SpecSub/DEMUCS',
    'Sigma given to SURE SGMSE',
    'Mean oSNR SpecSub',
    'Standard deviation oSNR SpecSub',
    'Uncertainty oSNR SpecSub',
    'Mean SURE threshold',
    'Std SURE threshold',
    'Uncertainty SURE threshold',
    'Mean oSNR DEMUCS',
    'Standard deviation oSNR DEMUCS',
    'Uncertainty oSNR DEMUCS',
    'Mean oSNR SGMSE',
    'Standard deviation oSNR SGMSE',
    'Uncertainty oSNR SGMSE',
    'Mean true sigma',
    'Std true sigma',
    'Uncertainty true sigma',
    'Mean UNSURE sigma',
    'Std UNSURE sigma',
    'Uncertainty UNSURE sigma',
    'Number of extracts'
])

for isnr in params['isnr']:

    oSNR_array = np.array(all_oSNR[isnr])
    oSNR_array_demucs = np.array(all_oSNR_demucs[isnr])
    oSNR_array_sgmse = np.array(all_oSNR_sgmse[isnr])
    sure_thresholds_array = np.array(all_sure_thresholds[isnr])
    sigma_true_array = np.array(all_sigma_true[isnr])
    sigma_estimate_array = np.array(all_sigma_estimate[isnr])

    sigma_array_specdemucs = np.array(params["sigma_specdemucs"])
    sigma_array_sgmse = np.array(params["sigma_sgmse"])

    N = len(sigma_true_array)

    mean_oSNR = np.mean(oSNR_array, axis=0)

    if N > 1:
        std_oSNR = np.std(oSNR_array, axis=0, ddof=1)
    else:
        std_oSNR = np.zeros(len(mean_oSNR))

    uncertainty = std_oSNR / np.sqrt(N)

    mean_oSNR_demucs = np.mean(oSNR_array_demucs, axis=0)

    if N > 1:
        std_oSNR_demucs = np.std(oSNR_array_demucs, axis=0, ddof=1)
    else:
        std_oSNR_demucs = np.zeros(len(mean_oSNR_demucs))

    uncertainty_demucs = std_oSNR_demucs / np.sqrt(N)

    mean_oSNR_sgmse = np.mean(oSNR_array_sgmse, axis=0)

    if N > 1:
        std_oSNR_sgmse = np.std(oSNR_array_sgmse, axis=0, ddof=1)
    else:
        std_oSNR_sgmse = np.zeros(len(mean_oSNR_sgmse))

    uncertainty_sgmse = std_oSNR_sgmse / np.sqrt(10)

    mean_sure_threshold = np.mean(sure_thresholds_array, axis=0)

    if N > 1:
        std_sure_threshold = np.std(sure_thresholds_array, axis=0, ddof=1)
    else:
        std_sure_threshold = np.zeros(len(mean_sure_threshold))

    sure_uncertainty = std_sure_threshold / np.sqrt(N)

    mean_sigma_true = np.mean(sigma_true_array)

    if N > 1:
        std_sigma_true = np.std(sigma_true_array, ddof=1)
    else:
        std_sigma_true = 0

    sigma_true_uncertainty = std_sigma_true / np.sqrt(N)

    mean_sigma_estimate = np.mean(sigma_estimate_array)

    if N > 1:
        std_sigma_estimate = np.std(sigma_estimate_array, ddof=1)
    else:
        std_sigma_estimate = 0

    sigma_estimate_uncertainty = std_sigma_estimate / np.sqrt(N)

    mask_specdemucs = sigma_array_specdemucs > 0
    mask_sgmse = sigma_array_sgmse > 0

    sigma_min_for_spans = min(
        np.min(sigma_array_specdemucs[mask_specdemucs]),
        np.min(sigma_array_sgmse[mask_sgmse]),
    )

    plt.semilogx(
        sigma_array_specdemucs[mask_specdemucs],
        mean_oSNR[mask_specdemucs],
        label='Soustraction spectrale, iSNR = ' + str(isnr) + ' dB'
    )

    plt.fill_between(
        sigma_array_specdemucs[mask_specdemucs],
        mean_oSNR[mask_specdemucs] - uncertainty[mask_specdemucs],
        mean_oSNR[mask_specdemucs] + uncertainty[mask_specdemucs],
        alpha=0.2
    )

    plt.semilogx(
        sigma_array_specdemucs[mask_specdemucs],
        mean_oSNR_demucs[mask_specdemucs],
        label='DEMUCS, iSNR = ' + str(isnr) + ' dB'
    )

    plt.fill_between(
        sigma_array_specdemucs[mask_specdemucs],
        mean_oSNR_demucs[mask_specdemucs] - uncertainty_demucs[mask_specdemucs],
        mean_oSNR_demucs[mask_specdemucs] + uncertainty_demucs[mask_specdemucs],
        alpha=0.2
    )

    plt.semilogx(
        sigma_array_sgmse[mask_sgmse],
        mean_oSNR_sgmse[mask_sgmse],
        label='SGMSE, iSNR = ' + str(isnr) + ' dB'
    )

    plt.fill_between(
        sigma_array_sgmse[mask_sgmse],
        mean_oSNR_sgmse[mask_sgmse] - uncertainty_sgmse[mask_sgmse],
        mean_oSNR_sgmse[mask_sgmse] + uncertainty_sgmse[mask_sgmse],
        alpha=0.2
    )

    plt.axvline(
        mean_sigma_true,
        color='black',
        linestyle='--',
        linewidth=2,
        label='Mean true sigma, iSNR = ' + str(isnr) + ' dB'
    )

    plt.axvspan(
        max(mean_sigma_true - sigma_true_uncertainty, sigma_min_for_spans),
        mean_sigma_true + sigma_true_uncertainty,
        color='black',
        alpha=0.15
    )

    plt.axvline(
        mean_sigma_estimate,
        color='red',
        linestyle='--',
        linewidth=2,
        label='Mean UNSURE sigma, iSNR = ' + str(isnr) + ' dB'
    )

    plt.axvspan(
        max(mean_sigma_estimate - sigma_estimate_uncertainty, sigma_min_for_spans),
        mean_sigma_estimate + sigma_estimate_uncertainty,
        color='red',
        alpha=0.15
    )

    max_sigma_len = max(len(sigma_array_specdemucs), len(sigma_array_sgmse))

    for i in range(max_sigma_len):

        has_specdemucs = i < len(sigma_array_specdemucs)
        has_sgmse = i < len(sigma_array_sgmse)

        CSV_export.append([
            isnr,

            sigma_array_specdemucs[i] if has_specdemucs else np.nan,
            sigma_array_sgmse[i] if has_sgmse else np.nan,

            mean_oSNR[i] if has_specdemucs else np.nan,
            std_oSNR[i] if has_specdemucs else np.nan,
            uncertainty[i] if has_specdemucs else np.nan,

            mean_sure_threshold[i] if has_specdemucs else np.nan,
            std_sure_threshold[i] if has_specdemucs else np.nan,
            sure_uncertainty[i] if has_specdemucs else np.nan,

            mean_oSNR_demucs[i] if has_specdemucs else np.nan,
            std_oSNR_demucs[i] if has_specdemucs else np.nan,
            uncertainty_demucs[i] if has_specdemucs else np.nan,

            mean_oSNR_sgmse[i] if has_sgmse else np.nan,
            std_oSNR_sgmse[i] if has_sgmse else np.nan,
            uncertainty_sgmse[i] if has_sgmse else np.nan,

            mean_sigma_true,
            std_sigma_true,
            sigma_true_uncertainty,

            mean_sigma_estimate,
            std_sigma_estimate,
            sigma_estimate_uncertainty,

            N
        ])

plt.xlabel('Sigma estimation for SURE')
plt.ylabel('oSNR (dB)')
plt.title('SURE optimal oSNR for various denoising methods')
plt.grid(True, which='both')
plt.legend()
plt.savefig(epath + "/mean_oSNR_as_function_of_sigma_sure.jpg", dpi=300, bbox_inches="tight")
plt.show()

with open(epath + '/Mean oSNR as function of sigma SURE.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(CSV_export)