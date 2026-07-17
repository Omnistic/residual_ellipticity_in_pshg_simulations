import os, yaml
from tqdm import tqdm
from dataclasses import dataclass
import numpy as np
from numpy import sin, cos
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit, root
import zospy as zp

HWP_ONLY_CONFIG = 1
HWP_ANGLE_COMMENT = "hwp_angle"
HWP_RETARDANCE_COMMENT = "hwp_retardance"

QWP_ANGLE_COMMENT = "qwp_angle"
QWP_RETARDANCE_COMMENT = "qwp_retardance"

IDEAL_HWP_RETARDANCE_IN_WAVES = 0.5
AHWP10M980_RETARDANCE_IN_WAVES_AT_880 = 0.51597
AQWP10M980_RETARDANCE_IN_WAVES_AT_880 = 0.25798

ELLIPTICITY_MFE_OPERAND = 8
ANGLE_MFE_OPERAND = 6

LASER_FWHM_IN_NM = 20
LASER_CENTER_IN_NM = 880
NUM_WAVELENGTHS = 9

DICHROIC_ANGLE_COMMENT = "dichroic_angle"
DICHROIC_RETARDANCE_COMMENT = "dichroic_retardance"
DICHROIC_CENTER_RETARDANCE_IN_DEG = 12.1
DICHROIC_FW_SPREAD_IN_DEG = 50
DICHROIC_ANGLE_IN_DEG = 0

POLARIZER_ANGLE_COMMENT = "linear_polarizer_angle"

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

def monochromatic(oss):
    if not oss.MCE.SetCurrentConfiguration(HWP_ONLY_CONFIG):
        raise Exception("Failed to set configuration")
    
    hwp_angle = zp.functions.lde.find_surface_by_comment(oss.LDE, HWP_ANGLE_COMMENT)[0]
    hwp_retardance = zp.functions.lde.find_surface_by_comment(oss.LDE, HWP_RETARDANCE_COMMENT)[0]

    qwp_angle = zp.functions.lde.find_surface_by_comment(oss.LDE, QWP_ANGLE_COMMENT)[0]
    qwp_retardance = zp.functions.lde.find_surface_by_comment(oss.LDE, QWP_RETARDANCE_COMMENT)[0]

    if hwp_angle == [] or hwp_retardance == []:
        raise Exception("HWP surfaces not found")

    if qwp_angle == [] or qwp_retardance == []:
        raise Exception("QWP surfaces not found")

    qwp_retardance.Thickness = 0
    qwp_angle.Thickness = 0

    hwp_angles = np.linspace(0, -90, 91)

    ideal_ellipticity = []
    ideal_angle = []

    real_ellipticity = []
    real_angle = []

    compensated_ellipticity = []
    compensated_angle = []

    for angle in hwp_angles:
        hwp_retardance.Thickness = 360 * IDEAL_HWP_RETARDANCE_IN_WAVES
        hwp_angle.Thickness = angle
        oss.MFE.CalculateMeritFunction()
        ideal_ellipticity.append(oss.MFE.GetOperandAt(ELLIPTICITY_MFE_OPERAND).Value)
        ideal_angle.append(oss.MFE.GetOperandAt(ANGLE_MFE_OPERAND).Value)

        hwp_retardance.Thickness = 360 * AHWP10M980_RETARDANCE_IN_WAVES_AT_880
        oss.MFE.CalculateMeritFunction()    
        real_ellipticity.append(oss.MFE.GetOperandAt(ELLIPTICITY_MFE_OPERAND).Value)
        real_angle.append(oss.MFE.GetOperandAt(ANGLE_MFE_OPERAND).Value)

    hwp_retardance.Thickness = 360 * AHWP10M980_RETARDANCE_IN_WAVES_AT_880
    qwp_retardance.Thickness = 360 * AQWP10M980_RETARDANCE_IN_WAVES_AT_880

    opt = oss.Tools.OpenLocalOptimization()

    for angle in hwp_angles:
        hwp_angle.Thickness = angle
        opt.RunAndWaitForCompletion()
        if qwp_angle.Thickness % 180 < 0:
            qwp_angle.Thickness += 180 * ( qwp_angle.Thickness // 180 )
        elif qwp_angle.Thickness % 180 > 0:
            qwp_angle.Thickness -= 180 * ( qwp_angle.Thickness // 180 )
        oss.MFE.CalculateMeritFunction()
        compensated_ellipticity.append(oss.MFE.GetOperandAt(ELLIPTICITY_MFE_OPERAND).Value)
        compensated_angle.append(oss.MFE.GetOperandAt(ANGLE_MFE_OPERAND).Value)

    opt.Close()

    inds = np.argsort(ideal_angle)
    ideal_angle = np.array(ideal_angle)[inds]
    ideal_ellipticity = np.array(ideal_ellipticity)[inds]

    inds = np.argsort(real_angle)
    real_angle = np.array(real_angle)[inds]
    real_ellipticity = np.array(real_ellipticity)[inds]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ideal_angle,
        y=ideal_ellipticity,
        mode="lines",
        name="0.5𝜆",
        line=dict(
            width=2,
            color=COLORS[6]+", 1)"
        )
    ))
    fig.add_trace(go.Scatter(
        x=real_angle,
        y=real_ellipticity,
        mode="lines",
        name=f"{AHWP10M980_RETARDANCE_IN_WAVES_AT_880:.3f}𝜆",
        line=dict(
            width=2,
            color=COLORS[2]+", 1)"
        )
    ))
    fig.add_trace(go.Scatter(
        x=compensated_angle,
        y=compensated_ellipticity,
        mode="lines",
        name=f"{AHWP10M980_RETARDANCE_IN_WAVES_AT_880:.3f}𝜆 + {AQWP10M980_RETARDANCE_IN_WAVES_AT_880:.3f}𝜆",
        line=dict(
            width=2,
            color=COLORS[1]+", 1)"
        )
    ))
    fig.add_trace(go.Scatter(
        x=[0, 180],
        y=[np.amax(ideal_ellipticity), np.amax(ideal_ellipticity)],
        mode="lines",
        line=dict(
            width=3,
            dash="dash",
            color=COLORS[6]+", 0.3)"
        ),
        showlegend=False
    ))
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
    fig.write_image(r"monochromatic_simulation.pdf", width=500, height=400)

def polychromatic(oss):
    make_polychromatic(oss)

    hwp_angle = zp.functions.lde.find_surface_by_comment(oss.LDE, HWP_ANGLE_COMMENT)[0]
    hwp_retardance = zp.functions.lde.find_surface_by_comment(oss.LDE, HWP_RETARDANCE_COMMENT)[0]

    qwp_angle = zp.functions.lde.find_surface_by_comment(oss.LDE, QWP_ANGLE_COMMENT)[0]
    qwp_retardance = zp.functions.lde.find_surface_by_comment(oss.LDE, QWP_RETARDANCE_COMMENT)[0]
    hwp_retardance.Thickness = 360 * AHWP10M980_RETARDANCE_IN_WAVES_AT_880
    qwp_retardance.Thickness = 360 * AQWP10M980_RETARDANCE_IN_WAVES_AT_880

    dichroic_angle = zp.functions.lde.find_surface_by_comment(oss.LDE, DICHROIC_ANGLE_COMMENT)[0]
    dichroic_angle.Thickness = DICHROIC_ANGLE_IN_DEG

    a_angles = np.linspace(0, 359, 360)

    hwp_angles = np.linspace(0, 90, 19)
    # hwp_angles = np.linspace(0, 90, 91)
    # hwp_angles = np.linspace(0, 90, 4)

    qwp_angles = np.linspace(0, 180, 37)
    # qwp_angles = np.linspace(0, 180, 181)
    # qwp_angles = np.linspace(0, 180, 7)

    monochromatic_ellipticity_map = -np.ones((len(hwp_angles), len(qwp_angles)))
    polychromatic_ellipticity_map = -np.ones((len(hwp_angles), len(qwp_angles)))
    monochromatic_alpha_map = -np.ones((len(hwp_angles), len(qwp_angles)))
    polychromatic_alpha_map = -np.ones((len(hwp_angles), len(qwp_angles)))
    for ii, ha in enumerate(hwp_angles):
        for jj, qa in enumerate(qwp_angles):
            hwp_angle.Thickness = ha
            qwp_angle.Thickness = qa
            oss.MFE.CalculateMeritFunction()
            monochromatic_ellipticity_map[ii, jj] = oss.MFE.GetOperandAt(ELLIPTICITY_MFE_OPERAND).Value
            monochromatic_alpha_map[ii, jj] = oss.MFE.GetOperandAt(11).Value
            intensity = np.zeros((len(a_angles), NUM_WAVELENGTHS))
            for kk in range(NUM_WAVELENGTHS):
                oss.MFE.GetOperandAt(1).GetOperandCell(zp.constants.Editors.MFE.MeritColumn.Param1).IntegerValue = kk + 1
                oss.MFE.CalculateMeritFunction()
                e_max = oss.MFE.GetOperandAt(2).Value
                e_min = oss.MFE.GetOperandAt(3).Value
                a_max = oss.MFE.GetOperandAt(4).Value
                intensity[:, kk] = e_max**2 * np.cos(np.deg2rad(a_max - a_angles))**2 + e_min**2 * np.sin(np.deg2rad(a_max - a_angles))**2
                intensity[:, kk] *= oss.MCE.GetOperandAt(2).GetOperandCell(kk + 1).DoubleValue
            polychromatic_intensity = np.sum(intensity, axis=1)
            ee, _, a_max_poly, _, _ = compute_polarization_parameters(np.deg2rad(a_angles), polychromatic_intensity)
            polychromatic_ellipticity_map[ii, jj] = ee
            polychromatic_alpha_map[ii, jj] = np.rad2deg(a_max_poly)
    
    min_search_start = 70
    min_search_stop = 111
    sub_poly_ellipticity_map = polychromatic_ellipticity_map[:, min_search_start:min_search_stop]
    sub_mono_ellipticity_map = monochromatic_ellipticity_map[:, min_search_start:min_search_stop]
    sub_poly_alpha_map = polychromatic_alpha_map[:, min_search_start:min_search_stop]
    sub_mono_alpha_map = monochromatic_alpha_map[:, min_search_start:min_search_stop]

    min_poly_indices = np.argmin(sub_poly_ellipticity_map, axis=1)
    min_mono_indices = np.argmin(sub_mono_ellipticity_map, axis=1)

    fig = go.Figure()
    poly_x = sub_poly_alpha_map[np.arange(len(min_poly_indices)), min_poly_indices]
    poly_y = sub_poly_ellipticity_map[np.arange(len(min_poly_indices)), min_poly_indices]
    sort_idx = np.argsort(poly_x)
    fig.add_trace(go.Scatter(
        x=poly_x[sort_idx],
        y=poly_y[sort_idx],
        mode="lines",
        name="Polychromatic",
        line=dict(
            width=2,
            color="rgba(0, 158, 115, 1)"
        )
    ))
    mono_x = sub_mono_alpha_map[np.arange(len(min_mono_indices)), min_mono_indices]
    mono_y = sub_mono_ellipticity_map[np.arange(len(min_mono_indices)), min_mono_indices]
    sort_idx = np.argsort(mono_x)
    fig.add_trace(go.Scatter(
        x=mono_x[sort_idx],
        y=mono_y[sort_idx],
        mode="lines",
        name="Monochromatic",
        line=dict(
            width=2,
            color="rgba(204, 121, 167, 1)"
        )
    ))
    fig.update_xaxes(
        title_text="Relative Polarization Angle (deg)",
        title_font=dict(size=20),
        showgrid=True,
        tickfont=dict(size=16),
        tickmode="array",
        tickvals=[0, 45, 90, 135, 180],
        range=[0, 180]
    )
    fig.update_yaxes(
        title_text="Ellipticity (-)",
        title_font=dict(size=20),
        showgrid=True,
        tickfont=dict(size=16),
        tickmode="array",
        tickvals=[0, 0.05, 0.1, 0.15, 0.2],
        range=[0, 0.20]
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
    fig.write_image(r"mono_poly_plot.pdf", width=500, height=400)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        x_title="QWP Motor Angle (deg)",
        y_title="HWP Motor Angle (deg)",
        subplot_titles=("Monochromatic", "Polychromatic")
    )
    fig.add_trace(go.Heatmap(
        z=monochromatic_ellipticity_map,
        x=qwp_angles,
        y=hwp_angles,
        coloraxis="coloraxis"
    ), row=1, col=1)
    fig.add_trace(go.Heatmap(
        z=polychromatic_ellipticity_map,
        x=qwp_angles,
        y=hwp_angles,
        coloraxis="coloraxis"
    ), row=2, col=1)
    # fig.add_trace(go.Scatter(
    #     x=qwp_angles[min_mono_indices+min_search_start],
    #     y=hwp_angles,
    #     opacity=0.4,
    #     mode="lines",
    #     line=dict(
    #         color="black",
    #         width=1.5
    #     ),
    #     showlegend=False
    # ), row=1, col=1)
    # fig.add_trace(go.Scatter(
    #     x=qwp_angles[min_poly_indices+min_search_start],
    #     y=hwp_angles,
    #     opacity=0.4,
    #     mode="lines",
    #     line=dict(
    #         color="black",
    #         width=1.5
    #     ),
    #     showlegend=False
    # ), row=2, col=1)
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
    fig.write_image(r"mono_poly_map.pdf", width=500, height=400)

    return

def vary_dichroic_retardance(oss):
    hwp_angle = zp.functions.lde.find_surface_by_comment(oss.LDE, HWP_ANGLE_COMMENT)[0]
    hwp_retardance = zp.functions.lde.find_surface_by_comment(oss.LDE, HWP_RETARDANCE_COMMENT)[0]
    qwp_angle = zp.functions.lde.find_surface_by_comment(oss.LDE, QWP_ANGLE_COMMENT)[0]
    qwp_retardance = zp.functions.lde.find_surface_by_comment(oss.LDE, QWP_RETARDANCE_COMMENT)[0]
    hwp_retardance.Thickness = 180 # 360 * AHWP10M980_RETARDANCE_IN_WAVES_AT_880
    qwp_retardance.Thickness = 90 # 360 * AQWP10M980_RETARDANCE_IN_WAVES_AT_880

    dichroic_retardance = zp.functions.lde.find_surface_by_comment(oss.LDE, DICHROIC_RETARDANCE_COMMENT)[0]
    retardance_array = np.linspace(0, 180, 19)

    hwp_angles = np.linspace(0, 90, 19)
    qwp_angles = np.linspace(0, 180, 37)
    pol_angles = np.linspace(0, 359, 360)

    custom_colorscale = [
        [0.0, COLORS[5]+" 255)"],
        [0.5, "rgba(255,255,255, 255)"],
        [1.0, COLORS[4]+" 255)"]
    ]
    retardance_array = [70]
    for retardance in retardance_array:
        dichroic_retardance.Thickness = retardance
        ellipticity_map = -np.ones((len(hwp_angles), len(qwp_angles)))
        intensities = []

        for ii, ha in enumerate(hwp_angles):
            for jj, qa in enumerate(qwp_angles):
                hwp_angle.Thickness = ha
                qwp_angle.Thickness = qa
                oss.MFE.CalculateMeritFunction()
                ellipticity_map[ii, jj] = oss.MFE.GetOperandAt(ELLIPTICITY_MFE_OPERAND).Value

                e_max = oss.MFE.GetOperandAt(2).Value
                e_min = oss.MFE.GetOperandAt(3).Value
                a_max = oss.MFE.GetOperandAt(4).Value
                intensities.extend(e_max**2 * cos(np.deg2rad(a_max - pol_angles))**2 + e_min**2 * sin(np.deg2rad(a_max - pol_angles))**2)

        alpha_prime = np.tile(pol_angles, len(hwp_angles) * len(qwp_angles))
        phi_prime = np.tile(np.repeat(qwp_angles, len(pol_angles)), len(hwp_angles))
        theta_prime = np.repeat(hwp_angles, len(qwp_angles) * len(pol_angles))

        alpha_prime = alpha_prime.reshape(-1, 1).T
        phi_prime = phi_prime.reshape(-1, 1).T
        theta_prime = theta_prime.reshape(-1, 1).T

        primes = np.deg2rad(np.vstack((theta_prime, phi_prime, alpha_prime)))
        _, _, delta, theta_0, phi_0, _ = compute_system_parameters(primes, intensities)
        theta_motor = np.linspace(0, np.pi/2, 91)
        initial_guess = np.deg2rad([30, 125])
        phi_motor_solution_1, phi_motor_solution_2 = phi_motor_for_linear_polarization(theta_motor, theta_0, phi_0, delta, initial_guess=initial_guess)

        fig = go.Figure(data=go.Heatmap(
            z=ellipticity_map,
            x=qwp_angles,
            y=hwp_angles,
            coloraxis="coloraxis"
        ))
        fig.add_trace(go.Scatter(
            x=np.rad2deg(phi_motor_solution_1),
            y=np.rad2deg(theta_motor),
            mode="markers",
            marker=dict(color="black", width=1)
        ))
        fig.add_trace(go.Scatter(
            x=np.rad2deg(phi_motor_solution_2),
            y=np.rad2deg(theta_motor),
            mode="markers",
            marker=dict(color="black", width=1)
        ))
        fig.update_layout(
            coloraxis=dict(
                cmin=0,
                cmax=1,
                colorscale=custom_colorscale
            )
        )
        fig.show()

def make_polychromatic(oss):
    while oss.MCE.NumberOfConfigurations != NUM_WAVELENGTHS:
        if oss.MCE.NumberOfConfigurations < NUM_WAVELENGTHS:
            oss.MCE.AddConfiguration(False)
        else:
            oss.MCE.DeleteConfiguration(1)

    sigma = LASER_FWHM_IN_NM / (2 * np.sqrt(2 * np.log(2)))
    wl_in_nm = np.linspace(LASER_CENTER_IN_NM - 2*sigma, LASER_CENTER_IN_NM + 2*sigma, NUM_WAVELENGTHS)

    wl_op = oss.MCE.GetOperandAt(1)
    weight_op = oss.MCE.GetOperandAt(2)
    weights = []
    for ii, wl in enumerate(wl_in_nm):
        wl_op.GetOperandCell(ii+1).DoubleValue = wl/1000
        weights.append(np.exp(-0.5 * ((wl - LASER_CENTER_IN_NM) / sigma) ** 2))
    weights = np.array(weights)
    weights /= np.sum(weights)
    for ii, weight in enumerate(weights):
        weight_op.GetOperandCell(ii+1).DoubleValue = weight

    retardance = np.linspace(-DICHROIC_FW_SPREAD_IN_DEG/2+DICHROIC_CENTER_RETARDANCE_IN_DEG, DICHROIC_FW_SPREAD_IN_DEG/2+DICHROIC_CENTER_RETARDANCE_IN_DEG, NUM_WAVELENGTHS)

    ret_op = oss.MCE.GetOperandAt(3)
    for ii, ret in enumerate(retardance):
        ret_op.GetOperandCell(ii+1).DoubleValue = ret

    fine_wl = np.linspace(860, 900, 41)
    fine_weights = np.exp(-0.5 * ((fine_wl - LASER_CENTER_IN_NM) / sigma) ** 2)

    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Scatter(
        x=wl_in_nm,
        y=weights,
        mode="markers",
        marker=dict(
            size=10,
            symbol="circle-dot",
            color="rgba(0, 0, 0, 0)",
            line=dict(
                width=3,
                color=COLORS[6]+", 255)"
            )
        )
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=fine_wl,
        y=fine_weights*np.amax(weights),
        mode="lines",
        line=dict(
            width=1.5,
            dash="dash",
            color=COLORS[6]+", 255)"
        )
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=wl_in_nm,
        y=retardance,
        mode="lines+markers",
        line=dict(
            width=1.5,
            dash="dash",
            color=COLORS[2]+", 255)"
        ),
        marker=dict(
            size=10,
            symbol="circle-dot",
            color="rgba(0, 0, 0, 0)",
            line=dict(
                width=3,
                color=COLORS[2]+", 255)"
            )
        )
    ), row=1, col=2)
    fig.update_xaxes(
        title_text="Wavelength (nm)",
        title_font=dict(size=20),
        showgrid=True,
        tickfont=dict(size=16),
        range=[860, 900],
        tickmode="array",
        tickvals=[860, 870, 880, 890, 900]
    )
    fig.update_yaxes(
        title_text="Relative Weight (-)",
        title_font=dict(size=20),
        showgrid=True,
        range=[0, None],
        tickfont=dict(size=16),
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Retardance (deg)",
        title_font=dict(size=20),
        tickfont=dict(size=16),
        showgrid=True,
        row=1, col=2
    )
    fig.update_layout(
        width=1000,
        height=400,
        margin=dict(l=70, r=50, t=30, b=90),
        template="simple_white",
        font_family="crm12",
        showlegend=False
    )
    fig.show()
    fig.write_image(r"bandwidth_and_spectral_dependance.pdf", width=1000, height=400)

# def polarimeter_intensity(alpha: float, alpha_max: float, k: float, e_min: float) -> float:
#     e_max = e_min + k**2
#     return e_max**2 * cos(alpha_max - alpha)**2 + e_min**2 * sin(alpha_max - alpha)**2

# def compute_polarization_parameters(angles: np.ndarray, intensity: np.ndarray, fit_factor: float = 1E4, max_intensity: float = 10):
#     scaled_intensity = intensity * fit_factor
#     max_scaled_intensity = max_intensity * fit_factor

#     try:
#         popt, _ = curve_fit(
#             polarimeter_intensity, 
#             angles, 
#             scaled_intensity, 
#             bounds=((0, 0, 0), (np.pi, max_scaled_intensity, max_scaled_intensity))
#         )
#     except RuntimeError:
#         return -1, -1, None, np.inf

#     alpha_max, k, e_min = popt

#     e_max = k**2 + e_min
#     ellipticity = e_min / e_max

#     fitted_intensity = polarimeter_intensity(angles, *popt) / fit_factor

#     rmse = np.sqrt(np.mean((intensity - fitted_intensity) ** 2))
#     nrmse = rmse / np.mean(intensity)

#     return ellipticity, e_max, alpha_max, fitted_intensity, nrmse

# def compute_system_parameters(primes, aggregated_intensities):
#     popt, pcov, _, msg, _ = curve_fit(
#         general_intensity,
#         primes,
#         aggregated_intensities,
#         p0 = [1, 1, 10/180*np.pi, 10/180*np.pi, 10/180*np.pi, 10/180*np.pi],
#         bounds=([0, 0, 0, 0, 0, 0], [np.inf, np.inf, np.pi, np.pi/2, np.pi/2, np.pi]),
#         full_output=True
#     )

# def compute_system_parameters(primes, aggregated_intensities, n_restarts=15):
#     bounds = ([0, 0, -np.pi, 0, 0, 0], [np.inf, np.inf, np.pi, np.pi/2, np.pi/2, np.pi])
#     rng = np.random.default_rng()

#     best_popt, best_pcov, best_msg, best_resid = None, None, None, np.inf

#     for _ in range(n_restarts):
#         p0 = [
#             1,
#             rng.uniform(0.3, 2.0),
#             rng.uniform(-np.pi, np.pi),
#             rng.uniform(0, np.pi/2),
#             rng.uniform(0, np.pi/2),
#             rng.uniform(0, np.pi),
#         ]
#         try:
#             popt, pcov, _, msg, _ = curve_fit(
#                 general_intensity, primes, aggregated_intensities,
#                 p0=p0, bounds=bounds, full_output=True
#             )
#             resid = np.sum((general_intensity(primes, *popt) - aggregated_intensities) ** 2)
#             if resid < best_resid:
#                 best_popt, best_pcov, best_msg, best_resid = popt, pcov, msg, resid
#         except RuntimeError:
#             continue

#     popt, pcov, msg = best_popt, best_pcov, best_msg

#     # fit = general_intensity(primes, *popt)
#     # rmse = np.sqrt(np.mean((aggregated_intensities - fit) ** 2))

#     # print(pcov)
#     # print(msg)

#     # print(f"Intensity_0: {popt[0]:.5f}, Gamma: {popt[1]:.5f}, Delta: {np.rad2deg(popt[2]):.5f}, Theta_0: {np.rad2deg(popt[3]):.5f}, Phi_0: {np.rad2deg(popt[4]):.5f}, Alpha_0: {np.rad2deg(popt[5]):.5f}")

#     # fig = go.Figure(data=go.Scatter(x=np.arange(len(aggregated_intensities)), y=aggregated_intensities))
#     # fig.add_trace(go.Scatter(x=np.arange(len(aggregated_intensities)), y=fit))
#     # fig.show()

#     intensity_0 = popt[0]
#     gamma = popt[1]
#     delta = popt[2]
#     theta_0 = popt[3]
#     phi_0 = popt[4]
#     alpha_0 = popt[5]

#     return intensity_0, gamma, delta, theta_0, phi_0, alpha_0

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

def load_parameters(params_file, oss):
    with open(params_file) as f:
        params = yaml.safe_load(f)

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

def hwp_and_qwp_scan(oss, params, desc, hwp_angles, qwp_angles, pol_angles, intensities_filename, overwrite_intensities=True):
    if overwrite_intensities or not os.path.exists(intensities_filename):
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

    ellipticity = np.empty((len(hwp_angles), len(qwp_angles)))

    for ha_ind, ha in enumerate(hwp_angles):
        for qa_ind, qa in enumerate(qwp_angles):
            el, _, _, _, _ = compute_polarization_parameters(np.deg2rad(pol_angles), polarization_analyzer_intensities[:, ha_ind, qa_ind])
            ellipticity[ha_ind, qa_ind] = el

    return ellipticity

def hwp_and_qwp_polychromatic_scan(oss, params, desc, hwp_angles, qwp_angles, pol_angles, intensities_filename, overwrite_intensities=True):
    if overwrite_intensities or not os.path.exists(intensities_filename):
        polarization_analyzer_intensities = np.empty((len(pol_angles), len(hwp_angles), len(qwp_angles), 5))
        total_iters = len(hwp_angles) * len(pol_angles) * len(qwp_angles)
        with tqdm(total=total_iters, leave=False, desc=desc) as pbar:
            for ha_ind, ha in enumerate(hwp_angles):
                params["hwp"]["angle_surface"].Thickness = ha
                for qa_ind, qa in enumerate(qwp_angles):
                    params["qwp"]["angle_surface"].Thickness = qa
                    for pa_ind, pa in enumerate(pol_angles):
                        params["pol"]["angle_surface"].Thickness = pa
                        oss.MFE.CalculateMeritFunction()
                        for ind in range(1, 5+1):
                            polarization_analyzer_intensities[pa_ind, ha_ind, qa_ind, ind-1] = oss.MFE.GetOperandAt(2*ind).Value
                        pbar.update(1)
        np.save(intensities_filename, polarization_analyzer_intensities)
    else:
        polarization_analyzer_intensities = np.load(intensities_filename)

    ellipticity = np.empty((len(hwp_angles), len(qwp_angles)))

    # polarization_analyzer_intensities[:, :, :, 0] *= 1.353352832366000E-001	
    # polarization_analyzer_intensities[:, :, :, 1] *= 6.065306597126000E-001	
    # polarization_analyzer_intensities[:, :, :, 3] *= 6.065306597126000E-001	
    # polarization_analyzer_intensities[:, :, :, 4] *= 1.353352832366000E-001

    for ha_ind, ha in enumerate(hwp_angles):
        for qa_ind, qa in enumerate(qwp_angles):
            # el, _, _, _, _ = compute_polarization_parameters(np.deg2rad(pol_angles), polarization_analyzer_intensities[:, ha_ind, qa_ind, 3])
            el, _, _, _, _ = compute_polarization_parameters(np.deg2rad(pol_angles), np.sum(polarization_analyzer_intensities[:, ha_ind, qa_ind, 0:5], axis=-1))
            ellipticity[ha_ind, qa_ind] = el

    return ellipticity

def make_polychromatic(oss, params):
    def gaussian(wavelength, center_wavelength, standard_deviation):
        return np.exp(-0.5 * ((wavelength - center_wavelength) / standard_deviation) ** 2)

    number_of_wavelengths = 5 # Max is 24 in OpticStudio!
    center_wavelength_in_nm = 880
    fwhm_bandwidth_in_nm = 12.5

    standard_deviation_in_nm = fwhm_bandwidth_in_nm / (2 * np.sqrt(2 * np.log(2)))
    wavelengths_in_nm = np.linspace(center_wavelength_in_nm-2*standard_deviation_in_nm, center_wavelength_in_nm+2*standard_deviation_in_nm, number_of_wavelengths)
    wavelengths_in_um = wavelengths_in_nm / 1000
    weights = gaussian(wavelengths_in_nm, center_wavelength_in_nm, standard_deviation_in_nm)

    center_retardance = 12.1
    half_width_retardance = 20

    retardances = np.linspace(center_retardance-half_width_retardance, center_retardance+half_width_retardance, number_of_wavelengths)

    oss.MCE.DeleteAllConfigurations()
    oss.MCE.DeleteAllRows()

    wave_operand = oss.MCE.GetOperandAt(1)
    wave_operand.ChangeType(zp.constants.Editors.MCE.MultiConfigOperandType.WAVE)
    wlwt_operand = oss.MCE.InsertNewOperandAt(2)
    wlwt_operand.ChangeType(zp.constants.Editors.MCE.MultiConfigOperandType.WLWT)
    dc_retardance_operand = oss.MCE.InsertNewOperandAt(3)
    dc_retardance_operand.ChangeType(zp.constants.Editors.MCE.MultiConfigOperandType.THIC)
    dc_retardance_operand.Param1 = params["dic"]["retardance_surface"].SurfaceNumber

    for ind, wavelength_in_um in enumerate(wavelengths_in_um):
        wave_operand.GetOperandCell(oss.MCE.NumberOfConfigurations).DoubleValue = wavelength_in_um
        wlwt_operand.GetOperandCell(oss.MCE.NumberOfConfigurations).DoubleValue = weights[ind]
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
    # fig.write_image("revised_fig_2b.pdf", width=500, height=400)

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


if __name__ == "__main__":
    # oss = connect_opticstudio("revised_monochromatic.zmx")

    # === Simulation 1-4 : Fit Variations === #
    # params = load_parameters("sim_1_4_params.yaml", oss)
    # sim = simulation_multi_map_fit(
    #     oss,
    #     params,
    #     sim_id=params["sim"]["five_mirrors_and_dichroic_real_waveplates"],
    #     n_runs=10,
    # )
    # print_multi_map_fit_results(sim, print_single_runs=True)
    # ======================================= #

    # === Figure 2b === #
    # params = load_parameters("fig_2b_params.yaml", oss)
    # figure_2b(oss, params, overwrite_intensities=True)
    # ================= #

    # === Figure 4 === #
    # params = load_parameters("fig_4_params.yaml", oss)
    # figure_4(oss, params, overwrite_intensities=True)
    # ================= #

    # oss.save()

    oss = connect_opticstudio("revised_polychromatic.zmx")
    
    params = load_parameters("fig_4_params.yaml", oss)
    make_polychromatic(oss, params) # Run once to setup the lens file
    test = hwp_and_qwp_polychromatic_scan(
        oss,
        params,
        "Polychromatic Scan",
        np.linspace(0, 90, params["hqp_size"][0]),
        np.linspace(0, 180, params["hqp_size"][1]),
        np.linspace(0, 359, params["hqp_size"][2]),
        "hwp_qwp_polychromatic_intensities.npy",
        overwrite_intensities=False
    )

    oss.save()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        x_title="QWP Motor Angle (deg)",
        y_title="HWP Motor Angle (deg)",
        subplot_titles=("Monochromatic", "Polychromatic")
    )
    fig.add_trace(go.Heatmap(
        z=test,
        x=np.linspace(0, 180, params["hqp_size"][1]),
        y=np.linspace(0, 90, params["hqp_size"][0]),
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


# if __name__ == "__main__":
#     zos = zp.ZOS()
#     oss = zos.connect()
#     oss.load("revised_monochromatic.zmx")
#     oss.UpdateMode = zp.constants.LensUpdateMode.None_
#     oss.TheApplication.ShowChangesInUI = False

#     # monochromatic(oss)
#     # polychromatic(oss)
#     # vary_dichroic_retardance(oss)

#     revised_simulation(oss)

#     oss.save()