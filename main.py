import os, yaml
from tqdm import tqdm
from dataclasses import dataclass, field
import numpy as np
from numpy import sin, cos
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit, root
import zospy as zp

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
class GroundTruthParam:
    mode: str = "rng"       # "fixed" or "rng"
    value: float = None     # used if mode == "fixed"
    low: float = None       # used if mode == "rng"
    high: float = None      # used if mode == "rng"

    def resolve(self, rng):
        if self.mode == "fixed":
            return self.value
        elif self.mode == "rng":
            return rng.uniform(self.low, self.high)
        else:
            raise ValueError(f"Unknown ground truth mode '{self.mode}'")

@dataclass
class GroundTruthConfig:
    theta_0: GroundTruthParam = field(default_factory=lambda: GroundTruthParam(mode="rng", low=0, high=90))
    phi_0: GroundTruthParam = field(default_factory=lambda: GroundTruthParam(mode="rng", low=0, high=90))
    alpha_0: GroundTruthParam = field(default_factory=lambda: GroundTruthParam(mode="rng", low=0, high=180))
    dic_retardance: GroundTruthParam = field(default_factory=lambda: GroundTruthParam(mode="rng", low=-10, high=20))

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

def simulation_single_map_fit(oss, params, sim_id=1, ground_truth: GroundTruthConfig = None):
    hqp_rng = np.random.default_rng(params["hqp_rng_seed"])
    dic_rng = np.random.default_rng(params["dic_rng_seed"])
    fit_rng = np.random.default_rng(params["fit_rng_seed"])

    oss.MCE.SetCurrentConfiguration(sim_id)

    if ground_truth is None:
        ground_truth = GroundTruthConfig()

    true_theta_0 = ground_truth.theta_0.resolve(hqp_rng)
    true_phi_0 = ground_truth.phi_0.resolve(hqp_rng)
    true_alpha_0 = ground_truth.alpha_0.resolve(hqp_rng)

    if sim_id == 1:
        true_dic_retardance = 0
    else:
        true_dic_retardance = ground_truth.dic_retardance.resolve(dic_rng)

    params["dic"]["retardance_mc_operand"].GetCellAt(sim_id).DoubleValue = true_dic_retardance

    hwp_angles, qwp_angles, pol_angles, primes = create_angle_arrays(params["hqp_size"])

    aggregated_intensities = []
    total_iters = len(hwp_angles) * len(qwp_angles) * len(pol_angles)
    with tqdm(total=total_iters, leave=True, desc=params["sim_desc"][sim_id], position=1) as pbar:
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

def simulation_multi_map_fit(oss, params, sim_id=1, n_runs=1, ground_truth: GroundTruthConfig = None):
    results_list = []

    for _ in tqdm(range(n_runs), desc="Runs", position=0, leave=True):
        results = simulation_single_map_fit(oss, params, sim_id=sim_id, ground_truth=ground_truth)

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

    lines = []
    lines.append(f"{title:=^{table_width}}")
    lines.append("")
    lines.append("|".join(f"{h:<{w}}" for h in headers))
    lines.append("+".join("-" * w for _ in range(n_cols)))
    lines.append("|".join(fmt_cell(v, w, precision) for v in data))
    lines.append("|".join(fmt_cell(v, w, precision) for v in ground_truth))
    lines.append("")

    for line in lines:
        print(line)

    with open("tab_XX.txt", "a") as f:
        f.write("\n".join(lines) + "\n")

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

    lines = []
    lines.append(f"{title:=^{table_width}}")
    lines.append("")
    lines.append("|".join(f"{h:<{w}}" for h in headers))
    lines.append("+".join("-" * w for _ in range(n_cols)))
    lines.append("|".join(fmt_cell(v, w, precision) for v in data))
    lines.append("")

    for line in lines:
        print(line)

    with open("tab_XX.txt", "a") as f:
        f.write("\n".join(lines) + "\n")

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

def phi_minimum_from_ellipticity_map(qwp_angles, ellipticity_map, polarization_angle_map, search_low=60, search_high=120):
    valid_qwp_indices = np.where((qwp_angles > search_low) & (qwp_angles < search_high))[0]
    roi = ellipticity_map[:, valid_qwp_indices]
    local_min_indices = np.argmin(roi, axis=1)
    min_el = np.min(roi, axis=1)
    min_indices = valid_qwp_indices[local_min_indices]

    polarization_angle_at_min = polarization_angle_map[np.arange(len(min_indices)), min_indices]

    return qwp_angles[min_indices], min_el, polarization_angle_at_min

def half_waveplate_scan(oss, params, desc, hwp_angles, pol_angles, intensities_filename, overwrite_intensities=True, optimize=False):
    if optimize:
        local_opt = oss.Tools.OpenLocalOptimization()

    if overwrite_intensities or not os.path.exists(intensities_filename):
        polarization_analyzer_intensities = np.empty((len(pol_angles), len(hwp_angles)))
        total_iters = len(hwp_angles) * len(pol_angles)
        with tqdm(total=total_iters, leave=True, desc=desc) as pbar:
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
        with tqdm(total=total_iters, leave=True, desc=desc) as pbar:
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

def ellipticity_map(hwp_angles, qwp_angles, pol_angles, intensities, ellipticity_filename, polarization_angle_filename, overwrite=False):
    if overwrite or not os.path.exists(ellipticity_filename):
        ellipticity = np.empty((len(hwp_angles), len(qwp_angles)))
        polarization_angle = np.empty((len(hwp_angles), len(qwp_angles)))

        if intensities.ndim == 4:
            intensities = np.sum(intensities, axis=-1)

        for ha_ind in range(len(hwp_angles)):
            for qa_ind in range(len(qwp_angles)):
                el, _, aa, _, _ = compute_polarization_parameters(np.deg2rad(pol_angles), intensities[:, ha_ind, qa_ind])
                ellipticity[ha_ind, qa_ind] = el
                polarization_angle[ha_ind, qa_ind] = aa
        np.save(ellipticity_filename, ellipticity)
        np.save(polarization_angle_filename, polarization_angle)
    else:
        ellipticity = np.load(ellipticity_filename)
        polarization_angle = np.load(polarization_angle_filename)

    return ellipticity, polarization_angle

def hwp_and_qwp_polychromatic_scan(oss, params, desc, hwp_angles, qwp_angles, pol_angles, weights, intensities_filename, overwrite=True):
    if overwrite or not os.path.exists(intensities_filename):
        polarization_analyzer_intensities = np.empty((len(pol_angles), len(hwp_angles), len(qwp_angles), len(weights)))
        total_iters = len(hwp_angles) * len(pol_angles) * len(qwp_angles)
        with tqdm(total=total_iters, leave=True, desc=desc) as pbar:
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

    return polarization_analyzer_intensities

def compensated_ellipticity_from_fit(oss, params, desc, hwp_angles, qwp_angles, pol_angles, ellipticity_filename, polarization_angle_filename, overwrite=False):
    if overwrite or not os.path.exists(ellipticity_filename) or not os.path.exists(polarization_angle_filename):
        polarization_analyzer_intensities = np.empty((len(pol_angles), len(hwp_angles)))
        total_iters = len(hwp_angles) * len(pol_angles)
        with tqdm(total=total_iters, leave=True, desc=desc) as pbar:
            for ha_ind, ha in enumerate(hwp_angles):
                params["hwp"]["angle_surface"].Thickness = ha
                params["qwp"]["angle_surface"].Thickness = qwp_angles[ha_ind]
                for pa_ind, pa in enumerate(pol_angles):
                    params["pol"]["angle_surface"].Thickness = pa
                    polarization_analyzer_intensities[pa_ind, ha_ind] = oss.MFE.GetOperandValue(zp.constants.Editors.MFE.MeritOperandType.CODA, 0, 1, 0, 0, 0, 0, 0, 0)
                    pbar.update(1)

        ellipticity = np.empty((len(hwp_angles)))
        polarization_angle = np.empty((len(hwp_angles)))

        for ha_ind in range(len(hwp_angles)):
            el, _, aa, _, _ = compute_polarization_parameters(np.deg2rad(pol_angles), polarization_analyzer_intensities[:, ha_ind])
            ellipticity[ha_ind] = el
            polarization_angle[ha_ind] = aa

        np.save(ellipticity_filename, ellipticity)
        np.save(polarization_angle_filename, polarization_angle)
    else:
        ellipticity = np.load(ellipticity_filename)
        polarization_angle = np.load(polarization_angle_filename)

    return ellipticity, polarization_angle

def compensated_polychromatic_ellipticity_from_fit(oss, params, desc, hwp_angles, qwp_angles, pol_angles, weights, ellipticity_filename, polarization_angle_filename, overwrite=False):
    if overwrite or not os.path.exists(ellipticity_filename) or not os.path.exists(polarization_angle_filename):
        polarization_analyzer_intensities = np.empty((len(pol_angles), len(hwp_angles), len(weights)))
        total_iters = len(hwp_angles) * len(pol_angles)
        with tqdm(total=total_iters, leave=True, desc=desc) as pbar:
            for ha_ind, ha in enumerate(hwp_angles):
                params["hwp"]["angle_surface"].Thickness = ha
                params["qwp"]["angle_surface"].Thickness = qwp_angles[ha_ind]
                for pa_ind, pa in enumerate(pol_angles):
                    params["pol"]["angle_surface"].Thickness = pa
                    oss.MFE.CalculateMeritFunction()
                    for ind in range(1, len(weights)+1):
                        polarization_analyzer_intensities[pa_ind, ha_ind, ind-1] = oss.MFE.GetOperandAt(2*ind).Value * weights[ind-1]
                    pbar.update(1)

        ellipticity = np.empty((len(hwp_angles)))
        polarization_angle = np.empty((len(hwp_angles)))

        for ha_ind in range(len(hwp_angles)):
            el, _, aa, _, _ = compute_polarization_parameters(np.deg2rad(pol_angles), polarization_analyzer_intensities[:, ha_ind, :].sum(axis=1))
            ellipticity[ha_ind] = el
            polarization_angle[ha_ind] = aa

        np.save(ellipticity_filename, ellipticity)
        np.save(polarization_angle_filename, polarization_angle)
    else:
        ellipticity = np.load(ellipticity_filename)
        polarization_angle = np.load(polarization_angle_filename)

    return ellipticity, polarization_angle

def unwrap_periodic(x, period=180):
    x = np.asarray(x, dtype=float).copy()
    if len(x) > 1:
        diff0 = x[1] - x[0]
        if diff0 > period / 2:
            x[0] += period
        elif diff0 < -period / 2:
            x[0] -= period
    return np.unwrap(x, period=period)

def gaussian(wavelength, center_wavelength, standard_deviation):
    return np.exp(-0.5 * ((wavelength - center_wavelength) / standard_deviation) ** 2)

def make_polychromatic(oss, params, number_of_wavelengths, fwhm_bandwidth_in_nm=12.5, center_retardance=12.1, half_width_retardance=30):
    center_wavelength_in_nm = 880

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

def supplementary_figure_XX(params, overwrite=False):
    hwp_angles, qwp_angles, pol_angles, _ = create_angle_arrays(params["hqp_size"])
    
    # This is intended to work if the nominal number of wavelengths is 31!
    # Original data is real waveplates and dichroic retardance 12.1-20:12.1+20deg, 880nm center wavelength, 12.5nm FWHM bandwidth
    if overwrite or not os.path.exists("sfig_XX_ellipticity_maps.npy"):
        intensities = np.load("hwp_qwp_polychromatic_intensities_31w.npy")
        weights = np.load("hwp_qwp_polychromatic_intensities_31w_weights.npy")

        ellipticity_maps = np.empty((len(hwp_angles), len(qwp_angles), 5))
        polarization_angle_maps = np.empty((len(hwp_angles), len(qwp_angles), 5))

        # Monochromatic
        intensities_monochromatic = intensities[:, :, :, 15] / weights[15]
        for ha_ind in range(len(hwp_angles)):
            for qa_ind in range(len(qwp_angles)):
                el, _, aa, _, _ = compute_polarization_parameters(np.deg2rad(pol_angles), intensities_monochromatic[:, ha_ind, qa_ind])
                ellipticity_maps[ha_ind, qa_ind, 0] = el
                polarization_angle_maps[ha_ind, qa_ind, 0] = aa

        # Full polychromatic (31 wavelengths)
        for ha_ind in range(len(hwp_angles)):
            for qa_ind in range(len(qwp_angles)):
                el, _, aa, _, _ = compute_polarization_parameters(np.deg2rad(pol_angles), np.sum(intensities[:, ha_ind, qa_ind, :], axis=-1))
                ellipticity_maps[ha_ind, qa_ind, 4] = el
                polarization_angle_maps[ha_ind, qa_ind, 4] = aa

        # Partial polychromatic (3, 7, 15 wavelengths), centered on index 15
        strides = [15, 5, 2]  # -> 3, 7, 15 wavelengths respectively

        for map_ind, stride in enumerate(strides, start=1):
            idx = np.arange(15 % stride, 31, stride)
            partial_intensities = np.sum(intensities[:, :, :, idx], axis=-1) / np.sum(weights[idx])

            for ha_ind in range(len(hwp_angles)):
                for qa_ind in range(len(qwp_angles)):
                    el, _, aa, _, _ = compute_polarization_parameters(np.deg2rad(pol_angles), partial_intensities[:, ha_ind, qa_ind])
                    ellipticity_maps[ha_ind, qa_ind, map_ind] = el
                    polarization_angle_maps[ha_ind, qa_ind, map_ind] = aa

        np.save("sfig_XX_ellipticity_maps.npy", ellipticity_maps)
        np.save("sfig_XX_polarization_angle_maps.npy", polarization_angle_maps)
    else:
        ellipticity_maps = np.load("sfig_XX_ellipticity_maps.npy")
        polarization_angle_maps = np.load("sfig_XX_polarization_angle_maps.npy")

    # RMSE relative to full polychromatic (31 wavelengths)
    reference = ellipticity_maps[:, :, 4]
    number_of_wavelengths = [1, 3, 7, 15, 31]

    rmse_values = np.empty(len(number_of_wavelengths))
    for i in range(len(number_of_wavelengths)):
        diff = ellipticity_maps[:, :, i] - reference
        rmse_values[i] = np.sqrt(np.mean(diff**2))

    min_qwp_angles = np.empty((len(hwp_angles), ellipticity_maps.shape[2]))
    min_el = np.empty((len(hwp_angles), ellipticity_maps.shape[2]))
    min_polarization_angles = np.empty((len(hwp_angles), ellipticity_maps.shape[2]))
    for i in range(ellipticity_maps.shape[2]):
        qwp_at_min, el_at_min, pol_angle_at_min = phi_minimum_from_ellipticity_map(
            qwp_angles, ellipticity_maps[:, :, i], polarization_angle_maps[:, :, i], search_low=60, search_high=120
        )
        min_qwp_angles[:, i] = qwp_at_min
        min_el[:, i] = el_at_min
        min_polarization_angles[:, i] = np.rad2deg(pol_angle_at_min)

    fig = make_subplots(rows=3, cols=1)
    fig.add_trace(go.Heatmap(z=ellipticity_maps[:, :, 4], x=qwp_angles, y=hwp_angles, coloraxis="coloraxis"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=min_qwp_angles[:, 4], y=hwp_angles,
        mode="lines", line=dict(color="rgba(0,114,178,1.0)", width=3, dash="dash"),
        name="Minimum Ellipticity (-)",
        legend="legend2",
    ), row=1, col=1)

    selected_colors = ["black", COLORS[5] + ", 1)", COLORS[2] + ", 1)", COLORS[6] + ", 1)", COLORS[4] + ", 1)"]
    selected_dashes = ["dot", "dashdot", "longdash", "solid", "dash"]

    for i in range(ellipticity_maps.shape[2]):
        label = f"N<sub>λ</sub> = {number_of_wavelengths[i]}"
        if number_of_wavelengths[i] == 15:
            label = f"<b>{label}</b>"

        order = np.argsort(min_polarization_angles[:, i])
        x_sorted = min_polarization_angles[order, i]
        y_sorted = min_el[order, i]

        fig.add_trace(go.Scatter(x=x_sorted, y=y_sorted, mode="lines", line=dict(color=selected_colors[i], width=3, dash=selected_dashes[i]), name=label), row=2, col=1)

    text=[
        f"<b>N<sub>λ</sub> = {n}</b>" if n == 15
        else f"N<sub>λ</sub> = {n}*" if n == 31
        else f"N<sub>λ</sub> = {n}"
        for n in number_of_wavelengths
    ]
    fig.add_trace(go.Scatter(
        x=number_of_wavelengths, y=rmse_values, mode="lines+markers+text",
        line=dict(color="gray", width=3),
        marker=dict(size=10, color=selected_colors),
        text=text,
        textposition=["top right"] * (len(number_of_wavelengths) - 1) + ["top left"],
        showlegend=False,
    ), row=3, col=1)

    fig.update_xaxes(title_text="QWP Motor Angle (deg)", range=[0, 180], tick0=0, dtick=30, row=1, col=1)
    fig.update_yaxes(title_text="HWP Motor Angle (deg)", range=[0, 90], tick0=0, dtick=30, row=1, col=1)
    fig.update_xaxes(title_text="Relative Polarization Angle (deg)", tick0=0, dtick=30, row=2, col=1)
    fig.update_yaxes(title_text="Minimum Ellipticity (-)", row=2, col=1)
    fig.update_xaxes(title_text="Number of wavelengths N<sub>λ</sub> (-)", row=3, col=1)
    fig.update_yaxes(
        title_text="Map RMSE (-) (×10<sup>-3</sup>)",
        range=[-0.001, 0.02],
        tickmode="array",
        tickvals=[0, 0.005, 0.010, 0.015, 0.020],
        ticktext=["0", "5", "10", "15", "20"],
        row=3, col=1,
    )

    row1_y0, row1_y1 = fig.layout.yaxis.domain
    row2_y0, row2_y1 = fig.layout.yaxis2.domain

    fig.add_annotation(
        text="Ellipticity (-)",
        xref="paper", yref="paper",
        x=1.08, y=row1_y1 + 0.005,
        xanchor="center", yanchor="bottom",
        showarrow=False,
        font=dict(size=20),
    )
    fig.update_layout(
        template="simple_white", font_family="crm12", width=1000, height=1200,
        legend=dict(y=(row2_y0 + row2_y1) / 2, yanchor="middle", x=1.02, xanchor="left", font=dict(size=20)),
        legend2=dict(y=row1_y1 + 0.005, yanchor="bottom", x=0.5, xanchor="center", orientation="h", font=dict(size=20)),
        font=dict(size=20),
        coloraxis=dict(
            cmin=0, cmax=1, colorscale=CUSTOM_COLORSCALE,
            colorbar=dict(
                lenmode="fraction", len=row1_y1 - row1_y0, y=(row1_y0 + row1_y1) / 2, yanchor="middle",
                thickness=15, title="", tickfont=dict(size=20),
                tickmode="array", tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1], ticktext=["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"],
            ),
        ),
    )
    fig.show()
    fig.write_image("sfig_XX.pdf", width=1000, height=1200)

def plot_ellipticity_comparison(
    qwp_angles,
    hwp_angles,
    hwp_angles_for_p_sol,
    datasets,
    colorscale,
    width=1000,
    font_size=20,
):
    n_rows = len(datasets)
    x_range_heatmap = [0, 180]
    x_ticks_heatmap = [0, 30, 60, 90, 120, 150, 180]
    x_range_curve = [0, 185]
    x_ticks_curve = [0, 30, 60, 90, 120, 150, 180]
    y_range = [0, 90]
    y_ticks = [0, 30, 60, 90]
    line_black = dict(color="rgba(0,0,0,1.0)", width=3)
    line_blue = dict(color="rgba(0,114,178,1.0)", width=3)
    tick_font = dict(size=font_size)

    panel_labels = ["Monochromatic + Ideal Waveplate", "Monochromatic + Real Waveplate", "Polychromatic + Ideal Waveplate", "Polychromatic + Real Waveplate"]

    fig = make_subplots(
        rows=n_rows, cols=2,
        column_widths=[0.5, 0.5],
        row_heights=[1 / n_rows] * n_rows,
        horizontal_spacing=0.1,
        vertical_spacing=0.08,
        shared_xaxes=True,
    )

    for i, data in enumerate(datasets):
        row = i + 1
        show_legend = (row == 1)

        # Ellipticity maps
        fig.add_trace(go.Heatmap(z=data["ellipticity"], x=qwp_angles, y=hwp_angles, coloraxis="coloraxis"), row=row, col=1)
        fig.add_trace(go.Scatter(x=data["p_sol"], y=hwp_angles_for_p_sol, mode="lines", line={**line_black, "dash": "dot"}, name="Fit", showlegend=show_legend), row=row, col=1)
        fig.add_trace(go.Scatter(x=data["p_min_qwp_ind"], y=hwp_angles, mode="lines", line={**line_blue, "dash": "dash"}, name="Min", showlegend=show_legend), row=row, col=1)

        # Compensated ellipticity curves
        p_sol_aa_unwrapped = unwrap_periodic(data["p_sol_aa"])
        fig.add_trace(go.Scatter(x=p_sol_aa_unwrapped, y=data["p_sol_el"], mode="lines", line=line_black, name="Fit", legend="legend2", showlegend=show_legend), row=row, col=2)

        p_min_aa_unwrapped = unwrap_periodic(data["p_min_aa"])
        fig.add_trace(go.Scatter(x=p_min_aa_unwrapped, y=data["p_min_el"], mode="lines", line=line_blue, name="Min", legend="legend2", showlegend=show_legend), row=row, col=2)

        fig.update_xaxes(range=x_range_heatmap, tickfont=tick_font, tickmode="array", tickvals=x_ticks_heatmap, constrain="domain", row=row, col=1)
        fig.update_yaxes(title_text="HWP Motor Angle (deg)" if row == 1 else None, title_font=tick_font, range=y_range, tickfont=tick_font, tickmode="array", tickvals=y_ticks, scaleanchor=f"x{2*row-1}", scaleratio=1, constrain="domain", row=row, col=1)
        fig.update_xaxes(range=x_range_curve, tickfont=tick_font, tickmode="array", tickvals=x_ticks_curve, row=row, col=2)
        fig.update_yaxes(title_text="Ellipticity (-)" if row == 1 else None, title_font=tick_font, range=[0, 0.2], tickfont=tick_font, side="right", row=row, col=2)

        fig.add_trace(
            go.Scatter(
                x=[166], y=[60],
                mode="markers",
                marker=dict(symbol="arrow-right", size=14, color="black"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row, col=1,
        )

        fig.add_annotation(
            xref=f"x{2*row}", yref=f"y{2*row}",
            x=90, y=0.15,
            xanchor="center", yanchor="middle",
            text=panel_labels[i],
            showarrow=False,
            font=dict(size=font_size - 2, color="black"),
            bgcolor="rgba(0,0,0,0.08)",
            bordercolor="rgba(0,0,0,0)",
            borderpad=4,
        )

    fig.update_xaxes(title_text="QWP Motor Angle (deg)", title_font=tick_font, row=n_rows, col=1)
    fig.update_xaxes(title_text="Relative Polarization Angle (deg)", title_font=tick_font, row=n_rows, col=2)

    margin = dict(l=80, r=80, t=100, b=80)
    plot_area_w = width - margin["l"] - margin["r"]
    col1_width_px = plot_area_w * 0.5
    row_height_px = col1_width_px * (y_range[1] - y_range[0]) / (x_range_heatmap[1] - x_range_heatmap[0])
    height = margin["t"] + margin["b"] + row_height_px * n_rows

    fig.update_layout(
        template="simple_white", font_family="crm12", width=width, height=height, margin=margin,
        legend=dict(x=0.14, y=1.02 + 40 / height, orientation="h", font=tick_font),
        legend2=dict(x=0.95, y=1.02 + 40 / height, xanchor="right", orientation="h", font=tick_font),
        coloraxis=dict(
            cmin=0, cmax=1, colorscale=colorscale,
            colorbar=dict(
                x=0.43, xanchor="left",
                y=0.923, yanchor="middle",
                lenmode="pixels", len=230, thickness=15,
                title="Ellipticity (-)", title_font=tick_font, tickfont=tick_font,
                tickmode="array", tickvals=[0, 0.5, 1], ticktext=["0.0", "0.5", "1.0"],
            ),
        ),
    )
    return fig

def new_figure_4(oss, params, overwrite=False):
    MONO_IDEAL_INTENSITIES_FILE = "fig_4_hwp_qwp_mono_ideal_intensities.npy"
    MONO_IDEAL_ELLIPTICITY_FILE = "fig_4_hwp_qwp_mono_ideal_ellipticity.npy"
    MONO_IDEAL_POLARIZATION_ANGLE_FILE = "fig_4_hwp_qwp_mono_ideal_polarization_angle.npy"
    MONO_IDEAL_SYSTEM_PARAMETERS_FILE = "fig_4_hwp_qwp_mono_ideal_system_parameters.npy"
    MONO_IDEAL_COMPENSATED_ELLIPTICITY_FILE = "fig_4_hwp_qwp_mono_ideal_compensated_ellipticity.npy"
    MONO_IDEAL_COMPENSATED_POLARIZATION_ANGLE_FILE = "fig_4_hwp_qwp_mono_ideal_compensated_polarization_angle.npy"
    MONO_REAL_INTENSITIES_FILE = "fig_4_hwp_qwp_mono_real_intensities.npy"
    MONO_REAL_ELLIPTICITY_FILE = "fig_4_hwp_qwp_mono_real_ellipticity.npy"
    MONO_REAL_POLARIZATION_ANGLE_FILE = "fig_4_hwp_qwp_mono_real_polarization_angle.npy"
    MONO_REAL_SYSTEM_PARAMETERS_FILE = "fig_4_hwp_qwp_mono_real_system_parameters.npy"
    MONO_REAL_COMPENSATED_ELLIPTICITY_FILE = "fig_4_hwp_qwp_mono_real_compensated_ellipticity.npy"
    MONO_REAL_COMPENSATED_POLARIZATION_ANGLE_FILE = "fig_4_hwp_qwp_mono_real_compensated_polarization_angle.npy"
    POLY_IDEAL_INTENSITIES_FILE = "fig_4_hwp_qwp_poly_ideal_intensities.npy"
    POLY_IDEAL_ELLIPTICITY_FILE = "fig_4_hwp_qwp_poly_ideal_ellipticity.npy"
    POLY_IDEAL_POLARIZATION_ANGLE_FILE = "fig_4_hwp_qwp_poly_ideal_polarization_angle.npy"
    POLY_IDEAL_SYSTEM_PARAMETERS_FILE = "fig_4_hwp_qwp_poly_ideal_system_parameters.npy"
    POLY_IDEAL_COMPENSATED_ELLIPTICITY_FILE = "fig_4_hwp_qwp_poly_ideal_compensated_ellipticity.npy"
    POLY_IDEAL_COMPENSATED_POLARIZATION_ANGLE_FILE = "fig_4_hwp_qwp_poly_ideal_compensated_polarization_angle.npy"
    POLY_REAL_INTENSITIES_FILE = "fig_4_hwp_qwp_poly_real_intensities.npy"
    POLY_REAL_ELLIPTICITY_FILE = "fig_4_hwp_qwp_poly_real_ellipticity.npy"
    POLY_REAL_POLARIZATION_ANGLE_FILE = "fig_4_hwp_qwp_poly_real_polarization_angle.npy"
    POLY_REAL_SYSTEM_PARAMETERS_FILE = "fig_4_hwp_qwp_poly_real_system_parameters.npy"
    POLY_REAL_COMPENSATED_ELLIPTICITY_FILE = "fig_4_hwp_qwp_poly_real_compensated_ellipticity.npy"
    POLY_REAL_COMPENSATED_POLARIZATION_ANGLE_FILE = "fig_4_hwp_qwp_poly_real_compensated_polarization_angle.npy"

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

    mono_ideal_ellipticity, mono_ideal_polarization_angle = ellipticity_map(
        hwp_angles,
        qwp_angles,
        pol_angles,
        mono_ideal_intensities,
        MONO_IDEAL_ELLIPTICITY_FILE,
        MONO_IDEAL_POLARIZATION_ANGLE_FILE,
        overwrite=overwrite,
    )

    if overwrite or not os.path.exists(MONO_IDEAL_SYSTEM_PARAMETERS_FILE):
        mono_ideal_i0, mono_ideal_g, mono_ideal_d, mono_ideal_t0, mono_ideal_p0, mono_ideal_a0 = compute_system_parameters(primes, mono_ideal_intensities.transpose(1, 2, 0).ravel(), rng=fit_rng)
        np.save(MONO_IDEAL_SYSTEM_PARAMETERS_FILE, (mono_ideal_i0, mono_ideal_g, mono_ideal_d, mono_ideal_t0, mono_ideal_p0, mono_ideal_a0))
    else:
        mono_ideal_i0, mono_ideal_g, mono_ideal_d, mono_ideal_t0, mono_ideal_p0, mono_ideal_a0 = np.load(MONO_IDEAL_SYSTEM_PARAMETERS_FILE)
    print(f"Fig 4A | Mono & Ideal Fit: I_0={mono_ideal_i0:.6f}, gamma={mono_ideal_g:.6f}, delta={np.rad2deg(mono_ideal_d):.6f}°, theta_0={np.rad2deg(mono_ideal_t0):.6f}°, phi_0={np.rad2deg(mono_ideal_p0):.6f}°, alpha_0={np.rad2deg(mono_ideal_a0):.6f}°")

    mono_ideal_p_sol_1, mono_ideal_p_sol_2 = phi_motor_for_linear_polarization(theta_motor=np.deg2rad(hwp_angles_for_p_sol), theta_0=mono_ideal_t0, phi_0=mono_ideal_p0, delta=mono_ideal_d, initial_guess=[np.deg2rad(90), np.deg2rad(0)])
    mono_ideal_p_sol_1 = np.rad2deg(mono_ideal_p_sol_1)
    mono_ideal_p_sol_2 = np.rad2deg(mono_ideal_p_sol_2)
    if np.abs(np.mean(mono_ideal_p_sol_1)-90) < np.abs(np.mean(mono_ideal_p_sol_2)-90):
        mono_ideal_p_sol = mono_ideal_p_sol_1
    else:
        mono_ideal_p_sol = mono_ideal_p_sol_2

    mono_ideal_p_sol_el, mono_ideal_p_sol_aa = compensated_ellipticity_from_fit(
        oss,
        params,
        "Monochromatic and Ideal Waveplates (b)",
        hwp_angles_for_p_sol,
        mono_ideal_p_sol,
        pol_angles,
        MONO_IDEAL_COMPENSATED_ELLIPTICITY_FILE,
        MONO_IDEAL_COMPENSATED_POLARIZATION_ANGLE_FILE,
        overwrite=overwrite,
    )

    mono_ideal_p_min_qwp_ind, mono_ideal_p_min_el, mono_ideal_p_min_aa = phi_minimum_from_ellipticity_map(qwp_angles, mono_ideal_ellipticity, mono_ideal_polarization_angle, search_low=60, search_high=120)

    # === Monochromatic and real waveplate === #
    params["hwp"]["retardance_surface"].Thickness = 185.7492
    params["qwp"]["retardance_surface"].Thickness = 92.87280

    wavelengths_in_um, weights = make_polychromatic(
        oss,
        params,
        number_of_wavelengths=1,
    )

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

    mono_real_ellipticity, mono_real_polarization_angle = ellipticity_map(
        hwp_angles,
        qwp_angles,
        pol_angles,
        mono_real_intensities,
        MONO_REAL_ELLIPTICITY_FILE,
        MONO_REAL_POLARIZATION_ANGLE_FILE,
        overwrite=overwrite,
    )

    if overwrite or not os.path.exists(MONO_REAL_SYSTEM_PARAMETERS_FILE):
        mono_real_i0, mono_real_g, mono_real_d, mono_real_t0, mono_real_p0, mono_real_a0 = compute_system_parameters(primes, mono_real_intensities.transpose(1, 2, 0).ravel(), rng=fit_rng)
        np.save(MONO_REAL_SYSTEM_PARAMETERS_FILE, (mono_real_i0, mono_real_g, mono_real_d, mono_real_t0, mono_real_p0, mono_real_a0))
    else:
        mono_real_i0, mono_real_g, mono_real_d, mono_real_t0, mono_real_p0, mono_real_a0 = np.load(MONO_REAL_SYSTEM_PARAMETERS_FILE)
    print(f"Fig 4C | Mono & Real Fit: I_0={mono_real_i0:.6f}, gamma={mono_real_g:.6f}, delta={np.rad2deg(mono_real_d):.6f}°, theta_0={np.rad2deg(mono_real_t0):.6f}°, phi_0={np.rad2deg(mono_real_p0):.6f}°, alpha_0={np.rad2deg(mono_real_a0):.6f}°")

    mono_real_p_sol_1, mono_real_p_sol_2 = phi_motor_for_linear_polarization(theta_motor=np.deg2rad(hwp_angles_for_p_sol), theta_0=mono_real_t0, phi_0=mono_real_p0, delta=mono_real_d, initial_guess=[np.deg2rad(90), np.deg2rad(0)])
    mono_real_p_sol_1 = np.rad2deg(mono_real_p_sol_1)
    mono_real_p_sol_2 = np.rad2deg(mono_real_p_sol_2)
    if np.abs(np.mean(mono_real_p_sol_1)-90) < np.abs(np.mean(mono_real_p_sol_2)-90):
        mono_real_p_sol = mono_real_p_sol_1
    else:
        mono_real_p_sol = mono_real_p_sol_2

    mono_real_p_sol_el, mono_real_p_sol_aa = compensated_ellipticity_from_fit(
        oss,
        params,
        "Monochromatic and Real Waveplates (d)",
        hwp_angles_for_p_sol,
        mono_real_p_sol,
        pol_angles,
        MONO_REAL_COMPENSATED_ELLIPTICITY_FILE,
        MONO_REAL_COMPENSATED_POLARIZATION_ANGLE_FILE,
        overwrite=overwrite,
    )

    mono_real_p_min_qwp_ind, mono_real_p_min_el, mono_real_p_min_aa = phi_minimum_from_ellipticity_map(qwp_angles, mono_real_ellipticity, mono_real_polarization_angle, search_low=60, search_high=120)

    # === Polychromatic and ideal waveplates === #
    params["hwp"]["retardance_surface"].Thickness = 180
    params["qwp"]["retardance_surface"].Thickness = 90

    wavelengths_in_um, weights = make_polychromatic(
        oss,
        params,
        number_of_wavelengths=15,
    )

    poly_ideal_intensities = hwp_and_qwp_polychromatic_scan(
        oss,
        params,
        "Polychromatic and Ideal Waveplates (e)",
        hwp_angles,
        qwp_angles,
        pol_angles,
        weights,
        POLY_IDEAL_INTENSITIES_FILE,
        overwrite=overwrite,
    )

    poly_ideal_ellipticity, poly_ideal_polarization_angle = ellipticity_map(
        hwp_angles,
        qwp_angles,
        pol_angles,
        poly_ideal_intensities,
        POLY_IDEAL_ELLIPTICITY_FILE,
        POLY_IDEAL_POLARIZATION_ANGLE_FILE,
        overwrite=overwrite,
    )

    if overwrite or not os.path.exists(POLY_IDEAL_SYSTEM_PARAMETERS_FILE):
        poly_ideal_i0, poly_ideal_g, poly_ideal_d, poly_ideal_t0, poly_ideal_p0, poly_ideal_a0 = compute_system_parameters(primes, np.sum(poly_ideal_intensities, axis=-1).transpose(1, 2, 0).ravel(), rng=fit_rng)
        np.save(POLY_IDEAL_SYSTEM_PARAMETERS_FILE, (poly_ideal_i0, poly_ideal_g, poly_ideal_d, poly_ideal_t0, poly_ideal_p0, poly_ideal_a0))
    else:
        poly_ideal_i0, poly_ideal_g, poly_ideal_d, poly_ideal_t0, poly_ideal_p0, poly_ideal_a0 = np.load(POLY_IDEAL_SYSTEM_PARAMETERS_FILE)
    print(f"Fig 4E | Poly & Ideal Fit: I_0={poly_ideal_i0:.6f}, gamma={poly_ideal_g:.6f}, delta={np.rad2deg(poly_ideal_d):.6f}°, theta_0={np.rad2deg(poly_ideal_t0):.6f}°, phi_0={np.rad2deg(poly_ideal_p0):.6f}°, alpha_0={np.rad2deg(poly_ideal_a0):.6f}°")

    poly_ideal_p_sol_1, poly_ideal_p_sol_2 = phi_motor_for_linear_polarization(theta_motor=np.deg2rad(hwp_angles_for_p_sol), theta_0=poly_ideal_t0, phi_0=poly_ideal_p0, delta=poly_ideal_d, initial_guess=[np.deg2rad(90), np.deg2rad(0)])
    poly_ideal_p_sol_1 = np.rad2deg(poly_ideal_p_sol_1)
    poly_ideal_p_sol_2 = np.rad2deg(poly_ideal_p_sol_2)
    if np.abs(np.mean(poly_ideal_p_sol_1)-90) < np.abs(np.mean(poly_ideal_p_sol_2)-90):
        poly_ideal_p_sol = poly_ideal_p_sol_1
    else:
        poly_ideal_p_sol = poly_ideal_p_sol_2

    poly_ideal_p_sol_el, poly_ideal_p_sol_aa = compensated_polychromatic_ellipticity_from_fit(
        oss,
        params,
        "Polychromatic and Ideal Waveplates (f)",
        hwp_angles_for_p_sol,
        poly_ideal_p_sol,
        pol_angles,
        weights,
        POLY_IDEAL_COMPENSATED_ELLIPTICITY_FILE,
        POLY_IDEAL_COMPENSATED_POLARIZATION_ANGLE_FILE,
        overwrite=overwrite,
    )

    poly_ideal_p_min_qwp_ind, poly_ideal_p_min_el, poly_ideal_p_min_aa = phi_minimum_from_ellipticity_map(qwp_angles, poly_ideal_ellipticity, poly_ideal_polarization_angle, search_low=60, search_high=120)

    # === Polychromatic and real waveplates === #
    params["hwp"]["retardance_surface"].Thickness = 185.7492
    params["qwp"]["retardance_surface"].Thickness = 92.87280

    wavelengths_in_um, weights = make_polychromatic(
        oss,
        params,
        number_of_wavelengths=15,
    )

    poly_real_intensities = hwp_and_qwp_polychromatic_scan(
        oss,
        params,
        "Polychromatic and Real Waveplates (g)",
        hwp_angles,
        qwp_angles,
        pol_angles,
        weights,
        POLY_REAL_INTENSITIES_FILE,
        overwrite=overwrite,
    )

    poly_real_ellipticity, poly_real_polarization_angle = ellipticity_map(
        hwp_angles,
        qwp_angles,
        pol_angles,
        poly_real_intensities,
        POLY_REAL_ELLIPTICITY_FILE,
        POLY_REAL_POLARIZATION_ANGLE_FILE,
        overwrite=overwrite,
    )

    if overwrite or not os.path.exists(POLY_REAL_SYSTEM_PARAMETERS_FILE):
        poly_real_i0, poly_real_g, poly_real_d, poly_real_t0, poly_real_p0, poly_real_a0 = compute_system_parameters(primes, np.sum(poly_real_intensities, axis=-1).transpose(1, 2, 0).ravel(), rng=fit_rng)
        np.save(POLY_REAL_SYSTEM_PARAMETERS_FILE, (poly_real_i0, poly_real_g, poly_real_d, poly_real_t0, poly_real_p0, poly_real_a0))
    else:
        poly_real_i0, poly_real_g, poly_real_d, poly_real_t0, poly_real_p0, poly_real_a0 = np.load(POLY_REAL_SYSTEM_PARAMETERS_FILE)
    print(f"Fig 4G | Poly & Real Fit: I_0={poly_real_i0:.6f}, gamma={poly_real_g:.6f}, delta={np.rad2deg(poly_real_d):.6f}°, theta_0={np.rad2deg(poly_real_t0):.6f}°, phi_0={np.rad2deg(poly_real_p0):.6f}°, alpha_0={np.rad2deg(poly_real_a0):.6f}°")

    poly_real_p_sol_1, poly_real_p_sol_2 = phi_motor_for_linear_polarization(theta_motor=np.deg2rad(hwp_angles_for_p_sol), theta_0=poly_real_t0, phi_0=poly_real_p0, delta=poly_real_d, initial_guess=[np.deg2rad(90), np.deg2rad(0)])
    poly_real_p_sol_1 = np.rad2deg(poly_real_p_sol_1)
    poly_real_p_sol_2 = np.rad2deg(poly_real_p_sol_2)
    if np.abs(np.mean(poly_real_p_sol_1)-90) < np.abs(np.mean(poly_real_p_sol_2)-90):
        poly_real_p_sol = poly_real_p_sol_1
    else:
        poly_real_p_sol = poly_real_p_sol_2

    poly_real_p_sol_el, poly_real_p_sol_aa = compensated_polychromatic_ellipticity_from_fit(
        oss,
        params,
        "Polychromatic and Real Waveplates (h)",
        hwp_angles_for_p_sol,
        poly_real_p_sol,
        pol_angles,
        weights,
        POLY_REAL_COMPENSATED_ELLIPTICITY_FILE,
        POLY_REAL_COMPENSATED_POLARIZATION_ANGLE_FILE,
        overwrite=overwrite,
    )

    poly_real_p_min_qwp_ind, poly_real_p_min_el, poly_real_p_min_aa = phi_minimum_from_ellipticity_map(qwp_angles, poly_real_ellipticity, poly_real_polarization_angle, search_low=60, search_high=120)

    # === Plotting === #
    mono_ideal_p_sol_aa = np.rad2deg(mono_ideal_p_sol_aa)
    mono_ideal_p_min_aa = np.rad2deg(mono_ideal_p_min_aa)
    mono_real_p_sol_aa = np.rad2deg(mono_real_p_sol_aa)
    mono_real_p_min_aa = np.rad2deg(mono_real_p_min_aa)
    poly_ideal_p_sol_aa = np.rad2deg(poly_ideal_p_sol_aa)
    poly_ideal_p_min_aa = np.rad2deg(poly_ideal_p_min_aa)
    poly_real_p_sol_aa = np.rad2deg(poly_real_p_sol_aa)
    poly_real_p_min_aa = np.rad2deg(poly_real_p_min_aa)

    mono_ideal = dict(
        ellipticity=mono_ideal_ellipticity,
        p_sol=mono_ideal_p_sol,
        p_min_qwp_ind=mono_ideal_p_min_qwp_ind,
        p_sol_el=mono_ideal_p_sol_el,
        p_sol_aa=mono_ideal_p_sol_aa,
        p_min_el=mono_ideal_p_min_el,
        p_min_aa=mono_ideal_p_min_aa,
    )
    mono_real = dict(
        ellipticity=mono_real_ellipticity,
        p_sol=mono_real_p_sol,
        p_min_qwp_ind=mono_real_p_min_qwp_ind,
        p_sol_el=mono_real_p_sol_el,
        p_sol_aa=mono_real_p_sol_aa,
        p_min_el=mono_real_p_min_el,
        p_min_aa=mono_real_p_min_aa,
    )
    poly_ideal = dict(
        ellipticity=poly_ideal_ellipticity,
        p_sol=poly_ideal_p_sol,
        p_min_qwp_ind=poly_ideal_p_min_qwp_ind,
        p_sol_el=poly_ideal_p_sol_el,
        p_sol_aa=poly_ideal_p_sol_aa,
        p_min_el=poly_ideal_p_min_el,
        p_min_aa=poly_ideal_p_min_aa,
    )
    poly_real = dict(
        ellipticity=poly_real_ellipticity,
        p_sol=poly_real_p_sol,
        p_min_qwp_ind=poly_real_p_min_qwp_ind,
        p_sol_el=poly_real_p_sol_el,
        p_sol_aa=poly_real_p_sol_aa,
        p_min_el=poly_real_p_min_el,
        p_min_aa=poly_real_p_min_aa,
    )
    fig = plot_ellipticity_comparison(qwp_angles, hwp_angles, hwp_angles_for_p_sol,
                                      [mono_ideal, mono_real, poly_ideal, poly_real], CUSTOM_COLORSCALE)
    fig.show()

def new_table_XX(oss, params):
    n_runs = 5
    gt_before = GroundTruthConfig(
        theta_0=GroundTruthParam(mode="fixed", value=18),
        phi_0=GroundTruthParam(mode="fixed", value=11),
        alpha_0=GroundTruthParam(mode="fixed", value=33),
        dic_retardance=GroundTruthParam(mode="fixed", value=28),
    )
    gt_after = GroundTruthConfig(
        theta_0=GroundTruthParam(mode="fixed", value=18),
        phi_0=GroundTruthParam(mode="fixed", value=11),
        alpha_0=GroundTruthParam(mode="fixed", value=33),
        dic_retardance=GroundTruthParam(mode="fixed", value=18),
    )

    # Mirrors only, shows retardance of mirrors (dichroic retardance is ignored!)
    sim = simulation_multi_map_fit(
        oss,
        params,
        sim_id=params["sim"]["five_mirrors_ideal_waveplates"],
        n_runs=n_runs,
        ground_truth=gt_before,
    )
    print_multi_map_fit_results(sim, print_single_runs=True)
    # Ideal waveplates, before and after, shows all parameters are fitted exactly!
    sim = simulation_multi_map_fit(
        oss,
        params,
        sim_id=params["sim"]["five_mirrors_and_dichroic_ideal_waveplates"],
        n_runs=n_runs,
        ground_truth=gt_before,
    )
    print_multi_map_fit_results(sim, print_single_runs=True)
    sim = simulation_multi_map_fit(
        oss,
        params,
        sim_id=params["sim"]["five_mirrors_and_dichroic_ideal_waveplates"],
        n_runs=n_runs,
        ground_truth=gt_after,
    )
    print_multi_map_fit_results(sim, print_single_runs=True)
    # Real waveplates, before and after, shows some parameters are fitted with significant error: theta_0 and alpha_0
    # phi_0 remains accurately fitted, and the system retardance is off by about 1 degree
    sim = simulation_multi_map_fit(
        oss,
        params,
        sim_id=params["sim"]["five_mirrors_and_dichroic_real_waveplates"],
        n_runs=n_runs,
        ground_truth=gt_before,
    )
    print_multi_map_fit_results(sim, print_single_runs=True)
    sim = simulation_multi_map_fit(
        oss,
        params,
        sim_id=params["sim"]["five_mirrors_and_dichroic_real_waveplates"],
        n_runs=n_runs,
        ground_truth=gt_after,
    )
    print_multi_map_fit_results(sim, print_single_runs=True)

def supplementary_figure_ZZ(oss, params, overwrite=False):
    dichroic_retardance_half_widths = [0, 10, 20, 30, 40, 50, 60]

    hwp_angles, qwp_angles, pol_angles, _ = create_angle_arrays(params["hqp_size"])

    # Real (manufacturer-specified) waveplate retardances, matching the experimental system
    params["hwp"]["retardance_surface"].Thickness = 185.7492
    params["qwp"]["retardance_surface"].Thickness = 92.87280

    results = []

    for drhw in dichroic_retardance_half_widths:
        tag = f"drhw{drhw}"
        intensities_file = f"sfig_ZZ_poly_real_intensities_{tag}.npy"
        ellipticity_file = f"sfig_ZZ_poly_real_ellipticity_{tag}.npy"
        polarization_angle_file = f"sfig_ZZ_poly_real_polarization_angle_{tag}.npy"

        wavelengths_in_um, weights = make_polychromatic(
            oss,
            params,
            number_of_wavelengths=3,
            center_retardance=0,
            half_width_retardance=drhw,
        )

        intensities = hwp_and_qwp_polychromatic_scan(
            oss, params, f"Polychromatic Real Waveplates (dichroic \u00b1{drhw} deg)",
            hwp_angles, qwp_angles, pol_angles, weights,
            intensities_file, overwrite=overwrite,
        )

        ellipticity, polarization_angle = ellipticity_map(
            hwp_angles, qwp_angles, pol_angles, intensities,
            ellipticity_file, polarization_angle_file, overwrite=overwrite,
        )

        p_min_qwp_ind, p_min_el, p_min_aa = phi_minimum_from_ellipticity_map(
            qwp_angles, ellipticity, polarization_angle, search_low=60, search_high=120
        )

        p_min_aa_deg = unwrap_periodic(np.rad2deg(p_min_aa))

        results.append(dict(half_width_retardance=drhw, p_min_aa=p_min_aa_deg, p_min_el=p_min_el))

    # === Plotting: stacked ellipticity profiles, styled like sfig_XX panel B === #
    selected_colors = ["black", COLORS[5] + ", 1)", COLORS[2] + ", 1)", COLORS[6] + ", 1)",
                       COLORS[4] + ", 1)", COLORS[0] + ", 1)", COLORS[1] + ", 1)"]
    selected_dashes = ["dot", "dashdot", "longdash", "solid", "dash", "dashdot", "dot"]

    fig = go.Figure()
    for i, res in enumerate(results):
        label = f"(12.1\u00b1{res['half_width_retardance']})\u00b0"

        order = np.argsort(res["p_min_aa"])
        x_sorted = res["p_min_aa"][order]
        y_sorted = res["p_min_el"][order]

        fig.add_trace(go.Scatter(
            x=x_sorted,
            y=y_sorted,
            mode="lines",
            line=dict(color=selected_colors[i], width=3, dash=selected_dashes[i]),
            name=label,
        ))

    fig.update_xaxes(
        title_text="Relative Polarization Angle (deg)",
        tick0=0, dtick=30,
        title_font=dict(size=20),
        tickfont=dict(size=16),
    )
    fig.update_yaxes(
        title_text="Minimum Ellipticity (-)",
        title_font=dict(size=20),
        tickfont=dict(size=16),
    )
    fig.update_layout(
        width=1000,
        height=500,
        template="simple_white",
        font_family="crm12",
        font=dict(size=20),
        legend=dict(title="DC Retardance", font=dict(size=16)),
        margin=dict(l=70, r=50, t=50, b=70),
    )
    fig.show()
    fig.write_image("sfig_ZZ.pdf", width=1000, height=500)

    return results

def supplementary_figure_WW(oss, params, overwrite=False):
    # (label, number_of_wavelengths, fwhm_bandwidth_in_nm)
    bandwidth_cases = [
        ("Monochromatic", 1, None),
        ("2 nm", 3, 2.0),
        ("12.5 nm", 3, 12.5),
        ("25 nm", 3, 25.0),
    ]
    center_retardance = 12.1
    # Fixed dichroic dispersion slope (deg/nm), chosen so that 12.5 nm FWHM
    # reproduces the original ±20 deg half-width spread used elsewhere.
    reference_fwhm_in_nm = 12.5
    reference_half_width_retardance = 20
    reference_std_in_nm = reference_fwhm_in_nm / (2 * np.sqrt(2 * np.log(2)))
    retardance_slope_per_nm = reference_half_width_retardance / (2 * reference_std_in_nm)

    hwp_angles, qwp_angles, pol_angles, _ = create_angle_arrays(params["hqp_size"])

    # Real (manufacturer-specified) waveplate retardances, matching the experimental system
    params["hwp"]["retardance_surface"].Thickness = 185.7492
    params["qwp"]["retardance_surface"].Thickness = 92.87280

    results = []

    for label, n_wavelengths, fwhm in bandwidth_cases:
        tag = label.replace(" ", "").replace(".", "p")
        intensities_file = f"sfig_WW_poly_real_intensities_{tag}.npy"
        ellipticity_file = f"sfig_WW_poly_real_ellipticity_{tag}.npy"
        polarization_angle_file = f"sfig_WW_poly_real_polarization_angle_{tag}.npy"

        if fwhm is None:
            half_width_retardance = 0  # unused (single wavelength -> center_retardance only)
        else:
            std_in_nm = fwhm / (2 * np.sqrt(2 * np.log(2)))
            half_width_retardance = retardance_slope_per_nm * 2 * std_in_nm

        make_polychromatic_kwargs = dict(
            oss=oss,
            params=params,
            number_of_wavelengths=n_wavelengths,
            center_retardance=center_retardance,
            half_width_retardance=half_width_retardance,
        )
        if fwhm is not None:
            make_polychromatic_kwargs["fwhm_bandwidth_in_nm"] = fwhm

        wavelengths_in_um, weights = make_polychromatic(**make_polychromatic_kwargs)

        intensities = hwp_and_qwp_polychromatic_scan(
            oss, params, f"Real Waveplates ({label})",
            hwp_angles, qwp_angles, pol_angles, weights,
            intensities_file, overwrite=overwrite,
        )

        ellipticity, polarization_angle = ellipticity_map(
            hwp_angles, qwp_angles, pol_angles, intensities,
            ellipticity_file, polarization_angle_file, overwrite=overwrite,
        )

        p_min_qwp_ind, p_min_el, p_min_aa = phi_minimum_from_ellipticity_map(
            qwp_angles, ellipticity, polarization_angle, search_low=60, search_high=120
        )

        p_min_aa_deg = unwrap_periodic(np.rad2deg(p_min_aa))

        results.append(dict(label=label, p_min_aa=p_min_aa_deg, p_min_el=p_min_el))

    # === Plotting: stacked ellipticity profiles, styled like sfig_XX panel B === #
    selected_colors = ["black", COLORS[2] + ", 1)", COLORS[1] + ", 1)", COLORS[5] + ", 1)"]
    selected_dashes = ["dot", "longdash", "solid", "dashdot"]

    fig = go.Figure()
    for i, res in enumerate(results):
        order = np.argsort(res["p_min_aa"])
        x_sorted = res["p_min_aa"][order]
        y_sorted = res["p_min_el"][order]

        fig.add_trace(go.Scatter(
            x=x_sorted,
            y=y_sorted,
            mode="lines",
            line=dict(color=selected_colors[i], width=3, dash=selected_dashes[i]),
            name=res["label"],
        ))

    fig.update_xaxes(
        title_text="Relative Polarization Angle (deg)",
        tick0=0, dtick=30,
        title_font=dict(size=20),
        tickfont=dict(size=16),
    )
    fig.update_yaxes(
        title_text="Minimum Ellipticity (-)",
        title_font=dict(size=20),
        tickfont=dict(size=16),
    )
    fig.update_layout(
        width=1000,
        height=500,
        template="simple_white",
        font_family="crm12",
        font=dict(size=20),
        legend=dict(title="Laser Bandwidth (FWHM)", font=dict(size=16)),
        margin=dict(l=70, r=50, t=50, b=70),
    )
    fig.show()
    fig.write_image("sfig_WW.pdf", width=1000, height=500)

    return results

if __name__ == "__main__":
    # === Table XX =================== #
    # oss = connect_opticstudio("revised_monochromatic.zmx")
    # params = load_parameters("tab_XX.yaml", oss)
    # new_table_XX(oss, params)
    # oss.save()
    # ================================ #

    ## === Figure 2b ================= ##
    # oss = connect_opticstudio("revised_monochromatic.zmx")
    # params = load_parameters("fig_2b_params.yaml", oss)
    # figure_2b(oss, params, overwrite_intensities=False)
    # oss.save()
    ## =============================== ##

    ## === New Figure 4 ============== ##
    oss = connect_opticstudio("revised_polychromatic.zmx")
    params = load_parameters("new_fig_4_params.yaml", oss)
    new_figure_4(oss, params, overwrite=True)
    oss.save()
    ## =============================== ##

    ## === Supplementary Figure XX === ##
    # params = load_parameters("sfig_XX_params.yaml")
    # supplementary_figure_XX(params)
    ## =============================== ##

    ## === Supplementary Figure ZZ === ##
    # oss = connect_opticstudio("revised_polychromatic.zmx")
    # params = load_parameters("sfig_ZZ_params.yaml", oss)
    # supplementary_figure_ZZ(oss, params, overwrite=False)
    # oss.save()
    ## =============================== ##

    ## === Supplementary Figure WW === ##
    # oss = connect_opticstudio("revised_polychromatic.zmx")
    # params = load_parameters("sfig_WW_params.yaml", oss)
    # supplementary_figure_WW(oss, params, overwrite=False)
    # oss.save()
    ## =============================== ##