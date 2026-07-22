import os, yaml
from tqdm import tqdm
from dataclasses import dataclass
import numpy as np
from numpy import sin, cos
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit, root
import zospy as zp

# HWP_ONLY_CONFIG = 1
# HWP_ANGLE_COMMENT = "hwp_angle"
# HWP_RETARDANCE_COMMENT = "hwp_retardance"

# QWP_ANGLE_COMMENT = "qwp_angle"
# QWP_RETARDANCE_COMMENT = "qwp_retardance"

# IDEAL_HWP_RETARDANCE_IN_WAVES = 0.5
# AHWP10M980_RETARDANCE_IN_WAVES_AT_880 = 0.51597
# AQWP10M980_RETARDANCE_IN_WAVES_AT_880 = 0.25798

# ELLIPTICITY_MFE_OPERAND = 8
# ANGLE_MFE_OPERAND = 6

# LASER_FWHM_IN_NM = 20
# LASER_CENTER_IN_NM = 880
# NUM_WAVELENGTHS = 9

# DICHROIC_ANGLE_COMMENT = "dichroic_angle"
# DICHROIC_RETARDANCE_COMMENT = "dichroic_retardance"
# DICHROIC_CENTER_RETARDANCE_IN_DEG = 12.1
# DICHROIC_FW_SPREAD_IN_DEG = 50
# DICHROIC_ANGLE_IN_DEG = 0

# POLARIZER_ANGLE_COMMENT = "linear_polarizer_angle"

COLORS = [
    "rgba(230, 159, 0",
    "rgba(86, 180, 233",
    "rgba(0, 158, 115",
    "rgba(240, 228, 66",
    "rgba(0, 114, 178",
    "rgba(213, 94, 0",
    "rgba(204, 121, 167"
]

CUSTOM_COLORSCALE = [
    [0.0, "rgba(204, 121, 167, 1)"],
    [0.5, "rgba(255,255,255, 1)"],
    [1.0, "rgba(0, 158, 115, 1)"]
]

@dataclass
class SimulationSingleMapResults:
    title: str
    intensity_0: float
    gamma: float
    delta: float
    theta_0: float
    phi_0: float
    alpha_0: float
    true_theta_0: float
    true_phi_0: float
    true_alpha_0: float
    true_dic_retardance: float
    theta_0_unwrapped: bool = False
    alpha_0_unwrapped: bool = False

def simulation_single_map_fit(oss, params, sim_id=1):
    hqp_rng = np.random.default_rng(params["hqp_rng_seed"])
    dic_rng = np.random.default_rng(params["dic_rng_seed"])
    fit_rng = np.random.default_rng(params["fit_rng_seed"])

    oss.MCE.SetCurrentConfiguration(sim_id)

    true_theta_0 = hqp_rng.uniform(0, 90)
    true_phi_0 = hqp_rng.uniform(0, 90)
    true_alpha_0 = hqp_rng.uniform(0, 180)
    if sim_id == 1:
        true_dic_retardance = 0
    else:    
        true_dic_retardance = dic_rng.uniform(-10, 20)

    # true_dic_retardance = 20

    params["dic"]["retardance_mc_operand"].GetCellAt(sim_id).DoubleValue = true_dic_retardance

    hwp_angles, qwp_angles, pol_angles, primes = create_angle_arrays(params["hqp_size"])

    aggregated_intensities = []
    total_iters = len(hwp_angles) * len(qwp_angles) * len(pol_angles)
    with tqdm(total=total_iters, leave=False, desc=params["sim_desc"][sim_id], position=1) as pbar:
        for ha in hwp_angles:
            params["hwp"]["angle_surface"].Thickness = ha - true_theta_0
            for qa in qwp_angles:
                params["qwp"]["angle_surface"].Thickness = qa - true_phi_0
                for pa in pol_angles:
                    params["pol"]["angle_surface"].Thickness = pa - true_alpha_0
                    aggregated_intensities.append(
                        oss.MFE.GetOperandValue(zp.constants.Editors.MFE.MeritOperandType.CODA, 0, 1, 0, 0, 0, 0, 0, 0)
                    )
                    pbar.update(1)

    intensity_0, gamma, delta, theta_0, phi_0, alpha_0 = compute_system_parameters(primes, aggregated_intensities, rng=fit_rng)

    results = SimulationSingleMapResults(
        title=params["sim_desc"][sim_id],
        intensity_0=float(intensity_0),
        gamma=float(gamma),
        delta=float(np.rad2deg(delta)),
        theta_0=float(np.rad2deg(theta_0)),
        phi_0=float(np.rad2deg(phi_0)),
        alpha_0=float(np.rad2deg(alpha_0)),
        true_theta_0=true_theta_0,
        true_phi_0=true_phi_0,
        true_alpha_0=true_alpha_0,
        true_dic_retardance=true_dic_retardance,
    )

    return results

def simulation_multi_map_fit(oss, params, sim_id=1, n_runs=1):
    results_list = []

    for _ in tqdm(range(n_runs), desc="Runs", position=0, leave=False):
        results = simulation_single_map_fit(oss, params, sim_id=sim_id)

        if abs(results.true_theta_0 - results.theta_0) > 45:
            results.theta_0_unwrapped = True
            if results.true_theta_0 > results.theta_0:
                results.theta_0 += 90
            else:
                results.theta_0 -= 90

        if abs(results.true_alpha_0 - results.alpha_0) > 90:
            results.alpha_0_unwrapped = True
            if results.true_alpha_0 > results.alpha_0:
                results.alpha_0 += 180
            else:
                results.alpha_0 -= 180

        results_list.append(results)

    return results_list

def load_parameters(params_file, oss=None):
    with open(params_file) as f:
        params = yaml.safe_load(f)

    if oss is not None:
        params["hwp"]["angle_surface"] = zp.functions.lde.find_surface_by_comment(oss.LDE, params["hwp"]["angle_comment"])[0]
        params["hwp"]["retardance_surface"] = zp.functions.lde.find_surface_by_comment(oss.LDE, params["hwp"]["retardance_comment"])[0]
        params["qwp"]["angle_surface"] = zp.functions.lde.find_surface_by_comment(oss.LDE, params["qwp"]["angle_comment"])[0]
        params["qwp"]["retardance_surface"] = zp.functions.lde.find_surface_by_comment(oss.LDE, params["qwp"]["retardance_comment"])[0]
        params["dic"]["retardance_surface"] = zp.functions.lde.find_surface_by_comment(oss.LDE, params["dic"]["retardance_comment"])[0]
        params["dic"]["retardance_mc_operand"] = oss.MCE.GetOperandAt(params["dic"]["retardance_mc_operand_row_id"])
        params["pol"]["angle_surface"] = zp.functions.lde.find_surface_by_comment(oss.LDE, params["pol"]["angle_comment"])[0]

    return params

def connect_opticstudio(lens_file):
    zos = zp.ZOS()
    oss = zos.connect()
    oss.load(lens_file)
    oss.UpdateMode = zp.constants.LensUpdateMode.None_
    oss.TheApplication.ShowChangesInUI = False

    return oss

def create_angle_arrays(hqp_size):
    hwp_angles = np.linspace(0, 90, hqp_size[0])
    qwp_angles = np.linspace(0, 180, hqp_size[1])
    pol_angles = np.linspace(0, 359, hqp_size[2])

    alpha_prime = np.tile(pol_angles, len(hwp_angles) * len(qwp_angles))
    phi_prime = np.tile(np.repeat(qwp_angles, len(pol_angles)), len(hwp_angles))
    theta_prime = np.repeat(hwp_angles, len(qwp_angles) * len(pol_angles))

    alpha_prime = alpha_prime.reshape(-1, 1).T
    phi_prime = phi_prime.reshape(-1, 1).T
    theta_prime = theta_prime.reshape(-1, 1).T

    primes = np.deg2rad(np.vstack((theta_prime, phi_prime, alpha_prime)))

    return hwp_angles, qwp_angles, pol_angles, primes

def print_single_map_fit_results(results: SimulationSingleMapResults):
    def fmt_cell(v, w, precision):
        if isinstance(v, (int, float)):
            return f"{v:<{w}.{precision}f}"
        return f"{str(v):<{w}}"

    headers = ["", "I_0", "gamma", "delta", "Dichroic Retardance", "theta_0", "phi_0", "alpha_0", "Theta Unwrapped?", "Alpha Unwrapped?"]
    data = ["Fitted", results.intensity_0, results.gamma, results.delta, "", results.theta_0, results.phi_0, results.alpha_0, results.theta_0_unwrapped, results.alpha_0_unwrapped]
    ground_truth = ["Ground Truth", "", "", "", results.true_dic_retardance, results.true_theta_0, results.true_phi_0, results.true_alpha_0, "", ""]

    w = 20
    precision = 6
    n_cols = len(headers)
    table_width = n_cols * w + (n_cols - 1)

    title = f" Simulation: {results.title} "
    print(f"{title:=^{table_width}}")
    print("")
    print("|".join(f"{h:<{w}}" for h in headers))
    print("+".join("-" * w for _ in range(n_cols)))
    print("|".join(fmt_cell(v, w, precision) for v in data))
    print("|".join(fmt_cell(v, w, precision) for v in ground_truth))
    print("")

def print_multi_map_fit_results(results_list, print_single_runs=False):
    def fmt_cell(v, w, precision):
        if isinstance(v, (int, float)):
            return f"{v:<{w}.{precision}f}"
        return f"{str(v):<{w}}"
    
    intensity_0_list = [results.intensity_0 for results in results_list]
    gamma_list = [results.gamma for results in results_list]
    delta_list = [results.delta for results in results_list]

    theta_0_error_list = [results.theta_0 - results.true_theta_0 for results in results_list]
    phi_0_error_list = [results.phi_0 - results.true_phi_0 for results in results_list]
    alpha_0_error_list = [results.alpha_0 - results.true_alpha_0 for results in results_list]

    headers = ["", "I_0 (mean ± std)", "Gamma (mean ± std)", "Delta (mean ± std)", "Dichroic Retardance", "Theta_0 Error (mean ± std)", "Phi_0 Error (mean ± std)", "Alpha_0 Error (mean ± std)"]
    data = ["Results", f"{np.mean(intensity_0_list):.6f} ± {np.std(intensity_0_list):.6f}", f"{np.mean(gamma_list):.6f} ± {np.std(gamma_list):.6f}", f"{np.mean(delta_list):.6f} ± {np.std(delta_list):.6f}", f"{results_list[0].true_dic_retardance:.6f}", f"{np.mean(theta_0_error_list):.6f} ± {np.std(theta_0_error_list):.6f}", f"{np.mean(phi_0_error_list):.6f} ± {np.std(phi_0_error_list):.6f}", f"{np.mean(alpha_0_error_list):.6f} ± {np.std(alpha_0_error_list):.6f}"]

    w = 30
    precision = 6
    n_cols = len(headers)
    table_width = n_cols * w + (n_cols - 1)

    title = f" Simulation: {results_list[0].title} (n_runs={len(results_list)})"
    print(f"{title:=^{table_width}}")
    print("")
    print("|".join(f"{h:<{w}}" for h in headers))
    print("+".join("-" * w for _ in range(n_cols)))
    print("|".join(fmt_cell(v, w, precision) for v in data))
    print("")

    if print_single_runs:
        for results in results_list:
            print_single_map_fit_results(results)

def general_intensity(primes, intensity_0, gamma, delta, theta_0, phi_0, alpha_0):
    theta_prime, phi_prime, alpha_prime = primes

    theta = theta_prime - theta_0
    phi = phi_prime - phi_0
    alpha = alpha_prime - alpha_0

    two_theta_minus_phi = 2*theta - phi

    d_1 = -gamma * ( cos(delta)*sin(phi)*sin(two_theta_minus_phi) + sin(delta)*cos(phi)*cos(two_theta_minus_phi) )
    d_2 = -gamma * ( sin(delta)*sin(phi)*sin(two_theta_minus_phi) - cos(delta)*cos(phi)*cos(two_theta_minus_phi) )
    d_3 = sin(phi)*cos(two_theta_minus_phi)
    d_4 = cos(phi)*sin(two_theta_minus_phi)

    return intensity_0 * ( (d_1**2 + d_2**2)*cos(alpha)**2 + (d_3**2 + d_4**2)*sin(alpha)**2 + 2*(d_1*d_3 + d_2*d_4)*sin(alpha)*cos(alpha) )

def compute_system_parameters(primes, aggregated_intensities, n_restarts=15, rng=None):
    bounds = ([0, 0, -np.pi, 0, 0, 0], [np.inf, np.inf, 0, np.pi/2, np.pi/2, np.pi])

    if rng is None:
        rng = np.random.default_rng()

    best_popt, best_pcov, best_msg, best_resid = None, None, None, np.inf

    for _ in range(n_restarts):
        p0 = [
            1,
            rng.uniform(0.3, 2.0),
            rng.uniform(-np.pi, 0),
            rng.uniform(0, np.pi/2),
            rng.uniform(0, np.pi/2),
            rng.uniform(0, np.pi),
        ]
        try:
            popt, pcov, _, msg, _ = curve_fit(
                general_intensity, primes, aggregated_intensities,
                p0=p0, bounds=bounds, full_output=True
            )
            resid = np.sum((general_intensity(primes, *popt) - aggregated_intensities) ** 2)
            if resid < best_resid:
                best_popt, best_pcov, best_msg, best_resid = popt, pcov, msg, resid
        except RuntimeError:
            continue

    popt, pcov, msg = best_popt, best_pcov, best_msg

    intensity_0 = popt[0]
    gamma = popt[1]
    delta = popt[2]
    theta_0 = popt[3]
    phi_0 = popt[4]
    alpha_0 = popt[5]

    return intensity_0, gamma, delta, theta_0, phi_0, alpha_0

def polarization_analyzer_intensity(alpha, alpha_max, k, e_min):
    e_max = e_min + k**2
    return e_max**2 * cos(alpha_max - alpha)**2 + e_min**2 * sin(alpha_max - alpha)**2

def compute_polarization_parameters(angles, intensity, fit_factor=10000, max_intensity=10):
    scaled_intensity = intensity * fit_factor
    max_scaled_intensity = max_intensity * fit_factor

    popt, _ = curve_fit(
        polarization_analyzer_intensity, 
        angles, 
        scaled_intensity, 
        bounds=((0, 0, 0), (np.pi, max_scaled_intensity, max_scaled_intensity))
    )

    alpha_max, k, e_min = popt

    e_max = k**2 + e_min
    ellipticity = e_min / e_max
    e_max /= fit_factor**0.5

    fitted_intensity = polarization_analyzer_intensity(angles, *popt) / fit_factor

    rmse = np.sqrt(np.mean((intensity - fitted_intensity) ** 2))
    nrmse = rmse / np.mean(intensity)

    return ellipticity, e_max, alpha_max, fitted_intensity, nrmse

def linear_polarization(phi, theta, delta):
    return np.tan(2*phi) + np.tan(delta) * np.sin(2 * (2*theta - phi))

def phi_motor_for_linear_polarization(theta_motor, theta_0, phi_0, delta, initial_guess=[0, 0]):
    theta = theta_motor - theta_0

    phi_motor_solution_1 = []
    phi_motor_solution_2 = []

    for t in theta:
        phi_solution_1 = root(linear_polarization, initial_guess[0], args=(t, delta), method="lm")["x"][0]
        phi_solution_2 = root(linear_polarization, initial_guess[1], args=(t, delta), method="lm")["x"][0]
        phi_motor_solution_1.append(phi_solution_1+phi_0)
        phi_motor_solution_2.append(phi_solution_2+phi_0)

    phi_motor_solution_1 = np.array(phi_motor_solution_1)
    phi_motor_solution_2 = np.array(phi_motor_solution_2)

    return phi_motor_solution_1, phi_motor_solution_2

def phi_minimum_from_ellipticity_map(qwp_angles, ellipticity_map, search_low=60, search_high=120):
    valid_qwp_indices = np.where((qwp_angles>search_low) & (qwp_angles<search_high))[0]
    roi = ellipticity_map[:, valid_qwp_indices]
    local_min_indices = np.argmin(roi, axis=1)
    min_el = np.min(roi, axis=1)
    min_indices = valid_qwp_indices[local_min_indices]

    return qwp_angles[min_indices], min_el

def half_waveplate_scan(oss, params, desc, hwp_angles, pol_angles, intensities_filename, overwrite_intensities=True, optimize=False):
    if optimize:
        local_opt = oss.Tools.OpenLocalOptimization()

    if overwrite_intensities or not os.path.exists(intensities_filename):
        polarization_analyzer_intensities = np.empty((len(pol_angles), len(hwp_angles)))
        total_iters = len(hwp_angles) * len(pol_angles)
        with tqdm(total=total_iters, leave=False, desc=desc) as pbar:
            for ha_ind, ha in enumerate(hwp_angles):
                params["hwp"]["angle_surface"].Thickness = ha
                for pa_ind, pa in enumerate(pol_angles):
                    params["pol"]["angle_surface"].Thickness = pa
                    if optimize:
                        local_opt.RunAndWaitForCompletion()
                    polarization_analyzer_intensities[pa_ind, ha_ind] = oss.MFE.GetOperandValue(zp.constants.Editors.MFE.MeritOperandType.CODA, 0, 1, 0, 0, 0, 0, 0, 0)
                    pbar.update(1)
        np.save(intensities_filename, polarization_analyzer_intensities)
    else:
        polarization_analyzer_intensities = np.load(intensities_filename)

    if optimize:
        local_opt.Close()

    ellipticity = []
    alpha_max = []
    for ha_ind, ha in enumerate(hwp_angles):
        el, _, am, _, _ = compute_polarization_parameters(np.deg2rad(pol_angles), polarization_analyzer_intensities[:, ha_ind])
        ellipticity.append(el)
        alpha_max.append(np.rad2deg(am))

    inds = np.argsort(alpha_max)
    alpha_max = np.array(alpha_max)[inds]
    ellipticity = np.array(ellipticity)[inds]

    return alpha_max, ellipticity

def hwp_and_qwp_scan(oss, params, desc, hwp_angles, qwp_angles, pol_angles, intensities_filename, overwrite=False):
    if overwrite or not os.path.exists(intensities_filename):
        polarization_analyzer_intensities = np.empty((len(pol_angles), len(hwp_angles), len(qwp_angles)))
        total_iters = len(hwp_angles) * len(pol_angles) * len(qwp_angles)
        with tqdm(total=total_iters, leave=False, desc=desc) as pbar:
            for ha_ind, ha in enumerate(hwp_angles):
                params["hwp"]["angle_surface"].Thickness = ha
                for qa_ind, qa in enumerate(qwp_angles):
                    params["qwp"]["angle_surface"].Thickness = qa
                    for pa_ind, pa in enumerate(pol_angles):
                        params["pol"]["angle_surface"].Thickness = pa
                        polarization_analyzer_intensities[pa_ind, ha_ind, qa_ind] = oss.MFE.GetOperandValue(zp.constants.Editors.MFE.MeritOperandType.CODA, 0, 1, 0, 0, 0, 0, 0, 0)
                        pbar.update(1)
        np.save(intensities_filename, polarization_analyzer_intensities)
    else:
        polarization_analyzer_intensities = np.load(intensities_filename)

    return polarization_analyzer_intensities

def ellipticity_map(hwp_angles, qwp_angles, pol_angles, intensities, ellipticity_filename, overwrite=False):
    if overwrite or not os.path.exists(ellipticity_filename):
        ellipticity = np.empty((len(hwp_angles), len(qwp_angles)))

        for ha_ind in range(len(hwp_angles)):
            for qa_ind in range(len(qwp_angles)):
                el, _, _, _, _ = compute_polarization_parameters(np.deg2rad(pol_angles), intensities[:, ha_ind, qa_ind])
                ellipticity[ha_ind, qa_ind] = el
        np.save(ellipticity_filename, ellipticity)
    else:
        ellipticity = np.load(ellipticity_filename)

    return ellipticity

def hwp_and_qwp_polychromatic_scan(oss, params, desc, hwp_angles, qwp_angles, pol_angles, weights, intensities_filename, overwrite_intensities=True):
    if overwrite_intensities or not os.path.exists(intensities_filename):
        polarization_analyzer_intensities = np.empty((len(pol_angles), len(hwp_angles), len(qwp_angles), len(weights)))
        total_iters = len(hwp_angles) * len(pol_angles) * len(qwp_angles)
        with tqdm(total=total_iters, leave=False, desc=desc) as pbar:
            for ha_ind, ha in enumerate(hwp_angles):
                params["hwp"]["angle_surface"].Thickness = ha
                for qa_ind, qa in enumerate(qwp_angles):
                    params["qwp"]["angle_surface"].Thickness = qa
                    for pa_ind, pa in enumerate(pol_angles):
                        params["pol"]["angle_surface"].Thickness = pa
                        oss.MFE.CalculateMeritFunction()
                        for ind in range(1, len(weights)+1):
                            polarization_analyzer_intensities[pa_ind, ha_ind, qa_ind, ind-1] = oss.MFE.GetOperandAt(2*ind).Value * weights[ind-1]
                        pbar.update(1)
        np.save(intensities_filename, polarization_analyzer_intensities)
    else:
        polarization_analyzer_intensities = np.load(intensities_filename)

    ellipticity = np.empty((len(hwp_angles), len(qwp_angles)))
    for ha_ind, ha in enumerate(hwp_angles):
        for qa_ind, qa in enumerate(qwp_angles):
            # el, _, _, _, _ = compute_polarization_parameters(np.deg2rad(pol_angles), polarization_analyzer_intensities[:, ha_ind, qa_ind, 4])
            el, _, _, _, _ = compute_polarization_parameters(np.deg2rad(pol_angles), np.sum(polarization_analyzer_intensities[:, ha_ind, qa_ind, :], axis=-1))
            ellipticity[ha_ind, qa_ind] = el

    return ellipticity

def compensated_ellipticity_from_fit(oss, params, desc, hwp_angles, qwp_angles, pol_angles, ellipticity_filename, overwrite=False):
    if overwrite or not os.path.exists(ellipticity_filename):
        polarization_analyzer_intensities = np.empty((len(pol_angles), len(hwp_angles)))
        total_iters = len(hwp_angles) * len(pol_angles)
        with tqdm(total=total_iters, leave=False, desc=desc) as pbar:
            for ha_ind, ha in enumerate(hwp_angles):
                params["hwp"]["angle_surface"].Thickness = ha
                params["qwp"]["angle_surface"].Thickness = qwp_angles[ha_ind]
                for pa_ind, pa in enumerate(pol_angles):
                    params["pol"]["angle_surface"].Thickness = pa
                    polarization_analyzer_intensities[pa_ind, ha_ind] = oss.MFE.GetOperandValue(zp.constants.Editors.MFE.MeritOperandType.CODA, 0, 1, 0, 0, 0, 0, 0, 0)
                    pbar.update(1)

        ellipticity = np.empty((len(hwp_angles)))

        for ha_ind in range(len(hwp_angles)):
                el, _, _, _, _ = compute_polarization_parameters(np.deg2rad(pol_angles), polarization_analyzer_intensities[:, ha_ind])
                ellipticity[ha_ind] = el
        np.save(ellipticity_filename, ellipticity)
    else:
        ellipticity = np.load(ellipticity_filename)

    return ellipticity

def make_polychromatic(oss, params, number_of_wavelengths):
    def gaussian(wavelength, center_wavelength, standard_deviation):
        return np.exp(-0.5 * ((wavelength - center_wavelength) / standard_deviation) ** 2)

    center_wavelength_in_nm = 880
    fwhm_bandwidth_in_nm = 12.5

    if number_of_wavelengths == 1:
        wavelengths_in_nm = np.array([center_wavelength_in_nm])
        wavelengths_in_um = wavelengths_in_nm / 1000
        weights = np.array([1.0])
    else:
        standard_deviation_in_nm = fwhm_bandwidth_in_nm / (2 * np.sqrt(2 * np.log(2)))
        wavelengths_in_nm = np.linspace(center_wavelength_in_nm-2*standard_deviation_in_nm, center_wavelength_in_nm+2*standard_deviation_in_nm, number_of_wavelengths)
        wavelengths_in_um = wavelengths_in_nm / 1000
        weights = gaussian(wavelengths_in_nm, center_wavelength_in_nm, standard_deviation_in_nm)
        weights /= np.sum(weights)

    center_retardance = 12.1
    half_width_retardance = 25
    if number_of_wavelengths == 1:
        retardances = np.array([center_retardance])
    else:
        retardances = np.linspace(center_retardance+half_width_retardance, center_retardance-half_width_retardance, number_of_wavelengths)

    oss.MCE.DeleteAllConfigurations()
    oss.MCE.DeleteAllRows()

    wave_operand = oss.MCE.GetOperandAt(1)
    wave_operand.ChangeType(zp.constants.Editors.MCE.MultiConfigOperandType.WAVE)
    dc_retardance_operand = oss.MCE.InsertNewOperandAt(2)
    dc_retardance_operand.ChangeType(zp.constants.Editors.MCE.MultiConfigOperandType.THIC)
    dc_retardance_operand.Param1 = params["dic"]["retardance_surface"].SurfaceNumber

    for ind, wavelength_in_um in enumerate(wavelengths_in_um):
        wave_operand.GetOperandCell(oss.MCE.NumberOfConfigurations).DoubleValue = wavelength_in_um
        dc_retardance_operand.GetOperandCell(oss.MCE.NumberOfConfigurations).DoubleValue = retardances[ind]
        oss.MCE.AddConfiguration(False)
    oss.MCE.DeleteConfiguration(oss.MCE.NumberOfConfigurations)

    oss.MFE.DeleteAllRows()

    for ind in range(wavelengths_in_um.size):
        op = oss.MFE.AddOperand()
        op.ChangeType(zp.constants.Editors.MFE.MeritOperandType.CONF)
        op.GetOperandCell(zp.constants.Editors.MFE.MeritColumn.Param1).IntegerValue = ind + 1
        op = oss.MFE.AddOperand()
        op.ChangeType(zp.constants.Editors.MFE.MeritOperandType.CODA)
    oss.MFE.DeleteRowAt(2)
    oss.MFE.DeleteRowAt(1)

    return wavelengths_in_um, weights

def figure_2b(oss, params, overwrite_intensities=True):
    hwp_angles = np.linspace(0, -90, params["hwp_only"]["size"])
    pol_angles = np.linspace(0, 359, params["polarizer"]["size"])

    oss.MCE.SetCurrentConfiguration(params["hwp_only"]["ideal_config"])
    hwp_only_ideal_wp_intensities_filename = "hwp_only_ideal_wp_intensities.npy"
    hwp_only_ideal_wp_alpha_max, hwp_only_ideal_wp_ellipticity = half_waveplate_scan(
        oss,
        params,
        params["hwp_only"]["ideal_desc"],
        hwp_angles,
        pol_angles,
        hwp_only_ideal_wp_intensities_filename,
        overwrite_intensities=overwrite_intensities,
        optimize=False,
    )

    oss.MCE.SetCurrentConfiguration(params["hwp_only"]["real_config"])
    hwp_only_real_wp_intensities_filename = "hwp_only_real_wp_intensities.npy"
    hwp_only_real_wp_alpha_max, hwp_only_real_wp_ellipticity = half_waveplate_scan(
        oss,
        params,
        params["hwp_only"]["real_desc"],
        hwp_angles,
        pol_angles,
        hwp_only_real_wp_intensities_filename,
        overwrite_intensities=overwrite_intensities,
        optimize=False,
    )

    oss.MCE.SetCurrentConfiguration(params["hwp_qwp"]["config"])
    hwp_qwp_real_wp_intensities_filename = "hwp_qwp_real_wp_intensities.npy"
    hwp_qwp_real_wp_alpha_max, hwp_qwp_real_wp_ellipticity = half_waveplate_scan(
        oss,
        params,
        params["hwp_qwp"]["desc"],
        hwp_angles,
        pol_angles,
        hwp_qwp_real_wp_intensities_filename,
        overwrite_intensities=overwrite_intensities,
        optimize=True,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hwp_only_ideal_wp_alpha_max,
        y=hwp_only_ideal_wp_ellipticity,
        mode="lines",
        name="0.5𝜆",
        line=dict(
            width=2,
            color=COLORS[6]+", 1)"
        )
    ))
    fig.add_trace(go.Scatter(
        x=hwp_only_real_wp_alpha_max,
        y=hwp_only_real_wp_ellipticity,
        mode="lines",
        name="0.516𝜆",
        line=dict(
            width=2,
            color=COLORS[2]+", 1)"
        )
    ))
    fig.add_trace(go.Scatter(
        x=hwp_qwp_real_wp_alpha_max,
        y=hwp_qwp_real_wp_ellipticity,
        mode="lines",
        name="0.516𝜆 + 0.258𝜆",
        line=dict(
            width=2,
            color=COLORS[1]+", 1)"
        )
    ))
    fig.add_trace(go.Scatter(
        x=[0, 180],
        y=[np.amax(hwp_only_ideal_wp_ellipticity), np.amax(hwp_only_ideal_wp_ellipticity)],
        mode="lines",
        line=dict(
            width=3,
            dash="dash",
            color=COLORS[6]+", 0.3)"
        ),
        showlegend=False
    ))
    fig.add_annotation(
        x=135,
        y=0.001,
        axref="x",
        ayref="y",
        ax=140,
        ay=0.04,
        arrowcolor=COLORS[1]+", 1)",
        arrowsize=1,
        arrowwidth=4,
        arrowhead=1,
    )
    fig.update_xaxes(
        title_text="Relative Polarization Angle (deg)",
        title_font=dict(size=20),
        showgrid=True,
        automargin=False,
        tickfont=dict(size=16),
        tickmode="array",
        tickvals=[0, 45, 90, 135, 180],
        range=[0, 180]
    )
    fig.update_yaxes(
        title_text="Ellipticity (-)",
        title_standoff=20,
        title_font=dict(size=20),
        showgrid=True,
        automargin=False,
        tickfont=dict(size=16),
        tickmode="array",
        tickvals=[0, 0.05, 0.1, 0.15, 0.2, 0.25],
        range=[0, 0.26]
    )
    fig.update_layout(
        width=500,
        height=400,
        margin=dict(l=70, r=50, t=50, b=70),
        template="simple_white",
        font_family="crm12",
        legend=dict(
            font=dict(size=16),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    fig.show()
    fig.write_image("revised_fig_2b.pdf", width=500, height=400)

def figure_4(oss, params, overwrite_intensities=True):
    hwp_angles = np.linspace(0, 90, params["hqp_size"][0])
    qwp_angles = np.linspace(0, 180, params["hqp_size"][1])
    pol_angles = np.linspace(0, 359, params["hqp_size"][2])

    oss.MCE.SetCurrentConfiguration(params["monochromatic"]["config"])
    hwp_qwp_monochromatic_intensities_filename = "hwp_qwp_monochromatic_intensities.npy"
    hwp_qwp_monochromatic_ellipticity = hwp_and_qwp_scan(
        oss,
        params,
        params["monochromatic"]["desc"],
        hwp_angles,
        qwp_angles,
        pol_angles,
        hwp_qwp_monochromatic_intensities_filename,
        overwrite_intensities=overwrite_intensities,
    )

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        x_title="QWP Motor Angle (deg)",
        y_title="HWP Motor Angle (deg)",
        subplot_titles=("Monochromatic", "Polychromatic")
    )
    fig.add_trace(go.Heatmap(
        z=hwp_qwp_monochromatic_ellipticity,
        x=qwp_angles,
        y=hwp_angles,
        coloraxis="coloraxis"
    ), row=1, col=1)
    fig.update_xaxes(
        tickmode="array",
        tickvals=[0, 30, 60, 90, 120, 150, 180],
        row=1, col=1
    )
    fig.update_xaxes(
        range=[0, 180],
        tickfont=dict(size=16),
        tickmode="array",
        tickvals=[0, 30, 60, 90, 120, 150, 180],
        row=2, col=1
    )
    fig.update_yaxes(
        range=[0, 90],
        tickfont=dict(size=16),
        tickmode="array",
        tickvals=[0, 30, 60, 90],
        row=1, col=1
    )
    fig.update_yaxes(
        range=[0, 90],
        tickfont=dict(size=16),
        tickmode="array",
        tickvals=[0, 30, 60, 90],
        row=2, col=1
    )
    fig.update_layout(
        width=500,
        height=400,
        margin=dict(l=70, r=50, t=50, b=70),
        template="simple_white",
        font_family="crm12",
        coloraxis=dict(
            cmin=0,
            cmax=1,
            colorscale=CUSTOM_COLORSCALE,
            colorbar_lenmode="pixels",
            colorbar_len=280,
            colorbar_thickness=15,
            colorbar_title="Ellipticity (-)",
            colorbar_title_font=dict(size=20),
            colorbar_tickfont=dict(size=16),
            colorbar_tickmode="array",
            colorbar_tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1],
            colorbar_ticktext=["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"],
        ),
        annotations=[
            dict(
                font=dict(size=20)
            ) for annotation in fig.layout.annotations
        ]
    )
    fig.show()

def supplementary_figure_XX(params):
    # This is intended to work if the nominal number of wavelengths is 31!
    intensities = np.load("hwp_qwp_polychromatic_intensities_31w.npy")
    weights = np.load("hwp_qwp_polychromatic_intensities_31w_weights.npy")

    fit_rng = np.random.default_rng(params["fit_rng_seed"])
    hwp_angles, qwp_angles, pol_angles, primes = create_angle_arrays(params["hqp_size"])

    intensities_monochromatic = intensities[:, :, :, 15] / weights[15]

    # intensities_monochromatic = np.load("hwp_qwp_polychromatic_intensities_1w_ideal.npy")[:,:, :, 0]

    if not os.path.exists("sfig_XX_monochromatic_fit_results.npy"):
        aggregated_intensities_monochromatic = intensities_monochromatic.transpose(1, 2, 0)
        aggregated_intensities_monochromatic = aggregated_intensities_monochromatic.ravel() 
        intensity_0, gamma, delta, theta_0, phi_0, alpha_0 = compute_system_parameters(primes, aggregated_intensities_monochromatic, rng=fit_rng)
        np.save("sfig_XX_monochromatic_fit_results.npy", np.array([intensity_0, gamma, delta, theta_0, phi_0, alpha_0]))
        print(f"Monochromatic Fit Results: I_0={intensity_0}, gamma={gamma}, delta={np.rad2deg(delta)}, theta_0={np.rad2deg(theta_0)}, phi_0={np.rad2deg(phi_0)}, alpha_0={np.rad2deg(alpha_0)}")
    else:
        intensity_0, gamma, delta, theta_0, phi_0, alpha_0 = np.load("sfig_XX_monochromatic_fit_results.npy")
        print(f"Monochromatic Fit Results (loaded): I_0={intensity_0}, gamma={gamma}, delta={np.rad2deg(delta)}, theta_0={np.rad2deg(theta_0)}, phi_0={np.rad2deg(phi_0)}, alpha_0={np.rad2deg(alpha_0)}")

    phi_motor_solution_1, phi_motor_solution_2 = phi_motor_for_linear_polarization(theta_motor=np.deg2rad(hwp_angles), theta_0=theta_0, phi_0=phi_0, delta=delta, initial_guess=[np.deg2rad(90), np.deg2rad(0)])
    if np.abs(np.mean(np.rad2deg(phi_motor_solution_1))-90) < np.abs(np.mean(np.rad2deg(phi_motor_solution_2))-90):
        phi_motor_plot = phi_motor_solution_1
    else:
        phi_motor_plot = phi_motor_solution_2

    monochromatic_ellipticity = np.empty((len(hwp_angles), len(qwp_angles)))
    phi_motor_min_search_low = 60
    phi_motor_min_search_high = 120
    phi_motor_min = np.empty(len(hwp_angles))
    phi_motor_min_el = np.empty(len(hwp_angles))
    for ha_ind in range(len(hwp_angles)):
        phi_min_el = np.inf
        for qa_ind in range(len(qwp_angles)):
            el, _, _, _, _ = compute_polarization_parameters(np.deg2rad(pol_angles), intensities_monochromatic[:, ha_ind, qa_ind])
            monochromatic_ellipticity[ha_ind, qa_ind] = el
            if qwp_angles[qa_ind] > phi_motor_min_search_low and qwp_angles[qa_ind] < phi_motor_min_search_high and el < phi_min_el:
                phi_min_el = el
                phi_motor_min[ha_ind] = qwp_angles[qa_ind]
                phi_motor_min_el[ha_ind] = el

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=monochromatic_ellipticity,
        x=np.linspace(0, 180, params["hqp_size"][1]),
        y=np.linspace(0, 90, params["hqp_size"][0]),
        coloraxis="coloraxis",
    ))       
    fig.add_trace(go.Scatter(
        x=np.rad2deg(phi_motor_plot),
        y=hwp_angles,
        mode="markers",
    ))  
    fig.add_trace(go.Scatter(
        x=phi_motor_min,
        y=hwp_angles,
        mode="markers",
    ))
    fig.update_layout(
        coloraxis=dict(
            cmin=0,
            cmax=1,
            colorscale=CUSTOM_COLORSCALE,
            colorbar_lenmode="pixels",
            colorbar_len=280,
            colorbar_thickness=15,
            colorbar_title="Ellipticity (-)",
            colorbar_title_font=dict(size=20),
            colorbar_tickfont=dict(size=16),
            colorbar_tickmode="array",
            colorbar_tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1],
            colorbar_ticktext=["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"],
        ),
    )
    fig.show()

def plot_ellipticity_comparison(
    qwp_angles,
    hwp_angles,
    hwp_angles_for_p_sol,
    ideal_data,
    real_data,
    colorscale,
    width=1000,
):
    x_range, y_range = [0, 180], [0, 90]
    x_ticks, y_ticks = [0, 30, 60, 90, 120, 150, 180], [0, 30, 60, 90]
    line_black = dict(color="rgba(0,0,0,1.0)", width=3)
    line_blue = dict(color="rgba(0,114,178,1.0)", width=3)
 
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.5, 0.5],
        row_heights=[0.5, 0.5],
        horizontal_spacing=0.1,
        vertical_spacing=0.15,
    )
 
    for row, data, show_legend in [(1, ideal_data, True), (2, real_data, False)]:
        # Ellipticity maps
        fig.add_trace(go.Heatmap(
            z=data["ellipticity"], x=qwp_angles, y=hwp_angles, coloraxis="coloraxis",
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=data["p_sol"], y=hwp_angles_for_p_sol, mode="lines",
            line={**line_black, "dash": "dot"},
            name="Fit", showlegend=show_legend,
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=data["p_min_qwp_ind"], y=hwp_angles, mode="lines",
            line={**line_blue, "dash": "dash"},
            name="Min", showlegend=show_legend,
        ), row=row, col=1)
 
        # Compensated ellipticity curves
        fig.add_trace(go.Scatter(
            x=hwp_angles_for_p_sol, y=data["p_sol_el"], mode="lines",
            line=line_black, name="Fit",
            legend="legend2", showlegend=show_legend,
        ), row=row, col=2)
        fig.add_trace(go.Scatter(
            x=hwp_angles, y=data["p_min_el"], mode="lines",
            line=line_blue, name="Min",
            legend="legend2", showlegend=show_legend,
        ), row=row, col=2)
 
        fig.update_xaxes(range=x_range, tickfont=dict(size=16),
                          tickmode="array", tickvals=x_ticks, row=row, col=1)
        fig.update_yaxes(range=y_range, tickfont=dict(size=16),
                          tickmode="array", tickvals=y_ticks,
                          scaleanchor="x1", scaleratio=1, constrain="domain",
                          row=row, col=1)
        
        fig.update_xaxes(range=[0, 90], tickfont=dict(size=16),
                         tickmode="array", tickvals=[0, 30, 60, 90], row=row, col=2)
        fig.update_yaxes(range=[0, 0.2], tickfont=dict(size=16), row=row, col=2)
 
    margin = dict(l=80, r=80, t=100, b=80)
    plot_area_w = width - margin["l"] - margin["r"]
    col1_width_px = plot_area_w * 0.5
    row1_height_px = col1_width_px * (y_range[1] - y_range[0]) / (x_range[1] - x_range[0])
    height = margin["t"] + margin["b"] + row1_height_px / 0.5
 
    fig.update_layout(
        template="simple_white",
        font_family="crm12",
        width=width,
        height=height,
        margin=margin,
        legend=dict(x=0.0, y=1.1, orientation="h", font=dict(size=16)),
        legend2=dict(x=1.0, y=1.1, xanchor="right", orientation="h", font=dict(size=16)),
        coloraxis=dict(
            cmin=0, cmax=1, colorscale=colorscale,
            colorbar_lenmode="pixels", colorbar_len=280, colorbar_thickness=15,
            colorbar_title="Ellipticity (-)",
            colorbar_title_font=dict(size=20),
            colorbar_tickfont=dict(size=16),
            colorbar_tickmode="array",
            colorbar_tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1],
            colorbar_ticktext=["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"],
        ),
    )
    return fig

def new_figure_4(oss, params, overwrite=False):
    MONO_IDEAL_INTENSITIES_FILE = "fig_4_hwp_qwp_mono_ideal_intensities.npy"
    MONO_IDEAL_ELLIPTICITY_FILE = "fig_4_hwp_qwp_mono_ideal_ellipticity.npy"
    MONO_IDEAL_COMPENSATED_ELLIPTICITY_FILE = "fig_4_hwp_qwp_mono_ideal_compensated_ellipticity.npy"
    MONO_REAL_INTENSITIES_FILE = "fig_4_hwp_qwp_mono_real_intensities.npy"
    MONO_REAL_ELLIPTICITY_FILE = "fig_4_hwp_qwp_mono_real_ellipticity.npy"
    MONO_REAL_COMPENSATED_ELLIPTICITY_FILE = "fig_4_hwp_qwp_mono_real_compensated_ellipticity.npy"

    hwp_angles, qwp_angles, pol_angles, primes = create_angle_arrays(params["hqp_size"])
    fit_rng = np.random.default_rng(params["fit_rng_seed"])
    hwp_angles_for_p_sol = np.linspace(0, 90, params["phi_sol_hwp"])

    # === Monochromatic and ideal waveplates === #
    params["hwp"]["retardance_surface"].Thickness = 180
    params["qwp"]["retardance_surface"].Thickness = 90

    wavelengths_in_um, weights = make_polychromatic(
        oss,
        params,
        number_of_wavelengths=1,
    )

    mono_ideal_intensities = hwp_and_qwp_scan(
        oss,
        params,
        "Monochromatic and Ideal Waveplates (a)",
        hwp_angles,
        qwp_angles,
        pol_angles,
        MONO_IDEAL_INTENSITIES_FILE,
        overwrite=overwrite,
    )

    mono_ideal_ellipticity = ellipticity_map(
        hwp_angles,
        qwp_angles,
        pol_angles,
        mono_ideal_intensities,
        MONO_IDEAL_ELLIPTICITY_FILE,
        overwrite=overwrite,
    )

    if overwrite or not os.path.exists("fig_4_hwp_qwp_mono_ideal_system_parameeters.npy"):
        mono_ideal_i0, mono_ideal_g, mono_ideal_d, mono_ideal_t0, mono_ideal_p0, mono_ideal_a0 = compute_system_parameters(primes, mono_ideal_intensities.transpose(1, 2, 0).ravel(), rng=fit_rng)
        np.save("fig_4_hwp_qwp_mono_ideal_system_parameeters.npy", (mono_ideal_i0, mono_ideal_g, mono_ideal_d, mono_ideal_t0, mono_ideal_p0, mono_ideal_a0))
    else:
        mono_ideal_i0, mono_ideal_g, mono_ideal_d, mono_ideal_t0, mono_ideal_p0, mono_ideal_a0 = np.load("fig_4_hwp_qwp_mono_ideal_system_parameeters.npy")
    print(f"Fig 4A | Mono & Ideal Fit: I_0={mono_ideal_i0:.6f}, gamma={mono_ideal_g:.6f}, delta={np.rad2deg(mono_ideal_d):.6f}°, theta_0={np.rad2deg(mono_ideal_t0):.6f}°, phi_0={np.rad2deg(mono_ideal_p0):.6f}°, alpha_0={np.rad2deg(mono_ideal_a0):.6f}°")

    mono_ideal_p_sol_1, mono_ideal_p_sol_2 = phi_motor_for_linear_polarization(theta_motor=np.deg2rad(hwp_angles_for_p_sol), theta_0=mono_ideal_t0, phi_0=mono_ideal_p0, delta=mono_ideal_d, initial_guess=[np.deg2rad(90), np.deg2rad(0)])
    mono_ideal_p_sol_1 = np.rad2deg(mono_ideal_p_sol_1)
    mono_ideal_p_sol_2 = np.rad2deg(mono_ideal_p_sol_2)
    if np.abs(np.mean(mono_ideal_p_sol_1)-90) < np.abs(np.mean(mono_ideal_p_sol_2)-90):
        mono_ideal_p_sol = mono_ideal_p_sol_1
    else:
        mono_ideal_p_sol = mono_ideal_p_sol_2

    mono_ideal_p_sol_el = compensated_ellipticity_from_fit(
        oss,
        params,
        "Monochromatic and Ideal Waveplates (b)",
        hwp_angles_for_p_sol,
        mono_ideal_p_sol,
        pol_angles,
        MONO_IDEAL_COMPENSATED_ELLIPTICITY_FILE,
        overwrite=overwrite,
    )

    mono_ideal_p_min_qwp_ind, mono_ideal_p_min_el = phi_minimum_from_ellipticity_map(qwp_angles, mono_ideal_ellipticity, search_low=60, search_high=120)

    # === Monochromatic and real waveplate === #
    params["hwp"]["retardance_surface"].Thickness = 185.7492
    params["qwp"]["retardance_surface"].Thickness = 92.87280

    mono_real_intensities = hwp_and_qwp_scan(
        oss,
        params,
        "Monochromatic and Real Waveplates (c)",
        hwp_angles,
        qwp_angles,
        pol_angles,
        MONO_REAL_INTENSITIES_FILE,
        overwrite=overwrite,
    )

    mono_real_ellipticity = ellipticity_map(
        hwp_angles,
        qwp_angles,
        pol_angles,
        mono_real_intensities,
        MONO_REAL_ELLIPTICITY_FILE,
        overwrite=overwrite,
    )

    if overwrite or not os.path.exists("fig_4_hwp_qwp_mono_real_system_parameeters.npy"):
        mono_real_i0, mono_real_g, mono_real_d, mono_real_t0, mono_real_p0, mono_real_a0 = compute_system_parameters(primes, mono_real_intensities.transpose(1, 2, 0).ravel(), rng=fit_rng)
        np.save("fig_4_hwp_qwp_mono_real_system_parameeters.npy", (mono_real_i0, mono_real_g, mono_real_d, mono_real_t0, mono_real_p0, mono_real_a0))
    else:
        mono_real_i0, mono_real_g, mono_real_d, mono_real_t0, mono_real_p0, mono_real_a0 = np.load("fig_4_hwp_qwp_mono_real_system_parameeters.npy")
    print(f"Fig 4C | Mono & Real Fit: I_0={mono_real_i0:.6f}, gamma={mono_real_g:.6f}, delta={np.rad2deg(mono_real_d):.6f}°, theta_0={np.rad2deg(mono_real_t0):.6f}°, phi_0={np.rad2deg(mono_real_p0):.6f}°, alpha_0={np.rad2deg(mono_real_a0):.6f}°")

    mono_real_p_sol_1, mono_real_p_sol_2 = phi_motor_for_linear_polarization(theta_motor=np.deg2rad(hwp_angles_for_p_sol), theta_0=mono_real_t0, phi_0=mono_real_p0, delta=mono_real_d, initial_guess=[np.deg2rad(90), np.deg2rad(0)])
    mono_real_p_sol_1 = np.rad2deg(mono_real_p_sol_1)
    mono_real_p_sol_2 = np.rad2deg(mono_real_p_sol_2)
    if np.abs(np.mean(mono_real_p_sol_1)-90) < np.abs(np.mean(mono_real_p_sol_2)-90):
        mono_real_p_sol = mono_real_p_sol_1
    else:
        mono_real_p_sol = mono_real_p_sol_2

    mono_real_p_sol_el = compensated_ellipticity_from_fit(
        oss,
        params,
        "Monochromatic and Real Waveplates (d)",
        hwp_angles_for_p_sol,
        mono_real_p_sol,
        pol_angles,
        MONO_REAL_COMPENSATED_ELLIPTICITY_FILE,
        overwrite=overwrite,
    )

    mono_real_p_min_qwp_ind, mono_real_p_min_el = phi_minimum_from_ellipticity_map(qwp_angles, mono_real_ellipticity, search_low=60, search_high=120)

    # === Plotting === #
    mono_ideal = dict(
        ellipticity=mono_ideal_ellipticity,
        p_sol=mono_ideal_p_sol,
        p_min_qwp_ind=mono_ideal_p_min_qwp_ind,
        p_sol_el=mono_ideal_p_sol_el,
        p_min_el=mono_ideal_p_min_el,
    )
    mono_real = dict(
        ellipticity=mono_real_ellipticity,
        p_sol=mono_real_p_sol,
        p_min_qwp_ind=mono_real_p_min_qwp_ind,
        p_sol_el=mono_real_p_sol_el,
        p_min_el=mono_real_p_min_el,
    )
    fig = plot_ellipticity_comparison(qwp_angles, hwp_angles, hwp_angles_for_p_sol,
                                    mono_ideal, mono_real, CUSTOM_COLORSCALE)
    fig.show()
    
if __name__ == "__main__":
    # oss = connect_opticstudio("revised_monochromatic.zmx")

    # === Simulation 1-4 : Fit Variations === #
    # params = load_parameters("sim_1_4_params.yaml", oss)
    # sim = simulation_multi_map_fit(
    #     oss,
    #     params,
    #     sim_id=params["sim"]["five_mirrors_and_dichroic_real_waveplates"],
    #     n_runs=1,
    # )
    # print_multi_map_fit_results(sim, print_single_runs=True)
    # ======================================= #

    # === Figure 2b === #
    # params = load_parameters("fig_2b_params.yaml", oss)
    # figure_2b(oss, params, overwrite_intensities=False)
    # ================= #

    # === Figure 4 === #
    # params = load_parameters("fig_4_params.yaml", oss)
    # figure_4(oss, params, overwrite_intensities=False)
    # ================= #

    # oss.save()



    oss = connect_opticstudio("revised_polychromatic.zmx")
    params = load_parameters("new_fig_4_params.yaml", oss)
    new_figure_4(oss, params, overwrite=False)
    oss.save()




    # Dichroic retardance = 12.1-20:12.1+20
    # params = load_parameters("sfig_XX_params.yaml")
    # supplementary_figure_XX(params)
