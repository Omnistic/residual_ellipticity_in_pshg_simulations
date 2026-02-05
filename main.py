import numpy as np
from numpy import sin, cos
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit, root
import zospy as zp

HWP_ONLY_CONFIG = 1
HWP_ANGLE_COMMENT = 'hwp_angle'
HWP_RETARDANCE_COMMENT = 'hwp_retardance'

QWP_ANGLE_COMMENT = 'qwp_angle'
QWP_RETARDANCE_COMMENT = 'qwp_retardance'

IDEAL_HWP_RETARDANCE_IN_WAVES = 0.5
AHWP10M980_RETARDANCE_IN_WAVES_AT_880 = 0.51597
AQWP10M980_RETARDANCE_IN_WAVES_AT_880 = 0.25798

ELLIPTICITY_MFE_OPERAND = 8
ANGLE_MFE_OPERAND = 6

LASER_FWHM_IN_NM = 20
LASER_CENTER_IN_NM = 880
NUM_WAVELENGTHS = 9

DICHROIC_ANGLE_COMMENT = 'dichroic_angle'
DICHROIC_RETARDANCE_COMMENT = 'dichroic_retardance'
DICHROIC_CENTER_RETARDANCE_IN_DEG = 12.1
DICHROIC_FW_SPREAD_IN_DEG = 50
DICHROIC_ANGLE_IN_DEG = 0

COLORS = [
    'rgba(230, 159, 0',
    'rgba(86, 180, 233',
    'rgba(0, 158, 115',
    'rgba(240, 228, 66',
    'rgba(0, 114, 178',
    'rgba(213, 94, 0',
    'rgba(204, 121, 167'
]

CUSTOM_COLORSCALE = [
    [0.0, 'rgba(204, 121, 167, 1)'],
    [0.5, 'rgba(255,255,255, 1)'],
    [1.0, 'rgba(0, 158, 115, 1)']
]

def monochromatic(oss):
    if not oss.MCE.SetCurrentConfiguration(HWP_ONLY_CONFIG):
        raise Exception('Failed to set configuration')
    
    hwp_angle = zp.functions.lde.find_surface_by_comment(oss.LDE, HWP_ANGLE_COMMENT)[0]
    hwp_retardance = zp.functions.lde.find_surface_by_comment(oss.LDE, HWP_RETARDANCE_COMMENT)[0]

    qwp_angle = zp.functions.lde.find_surface_by_comment(oss.LDE, QWP_ANGLE_COMMENT)[0]
    qwp_retardance = zp.functions.lde.find_surface_by_comment(oss.LDE, QWP_RETARDANCE_COMMENT)[0]

    if hwp_angle == [] or hwp_retardance == []:
        raise Exception('HWP surfaces not found')

    if qwp_angle == [] or qwp_retardance == []:
        raise Exception('QWP surfaces not found')

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
        mode='lines',
        name='0.5𝜆',
        line=dict(
            width=2,
            color=COLORS[6]+', 1)'
        )
    ))
    fig.add_trace(go.Scatter(
        x=real_angle,
        y=real_ellipticity,
        mode='lines',
        name=f'{AHWP10M980_RETARDANCE_IN_WAVES_AT_880:.3f}𝜆',
        line=dict(
            width=2,
            color=COLORS[2]+', 1)'
        )
    ))
    fig.add_trace(go.Scatter(
        x=compensated_angle,
        y=compensated_ellipticity,
        mode='lines',
        name=f'{AHWP10M980_RETARDANCE_IN_WAVES_AT_880:.3f}𝜆 + {AQWP10M980_RETARDANCE_IN_WAVES_AT_880:.3f}𝜆',
        line=dict(
            width=2,
            color=COLORS[1]+', 1)'
        )
    ))
    fig.add_trace(go.Scatter(
        x=[0, 180],
        y=[np.amax(ideal_ellipticity), np.amax(ideal_ellipticity)],
        mode='lines',
        line=dict(
            width=3,
            dash='dash',
            color=COLORS[6]+', 0.3)'
        ),
        showlegend=False
    ))
    fig.update_xaxes(
        title_text='Relative Polarization Angle (deg)',
        title_font=dict(size=20),
        showgrid=True,
        automargin=False,
        tickfont=dict(size=16),
        tickmode='array',
        tickvals=[0, 45, 90, 135, 180],
        range=[0, 180]
    )
    fig.update_yaxes(
        title_text='Ellipticity (-)',
        title_standoff=20,
        title_font=dict(size=20),
        showgrid=True,
        automargin=False,
        tickfont=dict(size=16),
        tickmode='array',
        tickvals=[0, 0.05, 0.1, 0.15, 0.2, 0.25],
        range=[0, 0.26]
    )
    fig.update_layout(
        width=500,
        height=400,
        margin=dict(l=70, r=50, t=50, b=70),
        template='simple_white',
        font_family='crm12',
        legend=dict(
            font=dict(size=16),
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    fig.show()
    fig.write_image(r'monochromatic_simulation.pdf', width=500, height=400)

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
    hwp_angles = np.linspace(0, 90, 91)
    # hwp_angles = np.linspace(0, 90, 4)

    qwp_angles = np.linspace(0, 180, 37)
    qwp_angles = np.linspace(0, 180, 181)
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
        mode='lines',
        name='Polychromatic',
        line=dict(
            width=2,
            color='rgba(0, 158, 115, 1)'
        )
    ))
    mono_x = sub_mono_alpha_map[np.arange(len(min_mono_indices)), min_mono_indices]
    mono_y = sub_mono_ellipticity_map[np.arange(len(min_mono_indices)), min_mono_indices]
    sort_idx = np.argsort(mono_x)
    fig.add_trace(go.Scatter(
        x=mono_x[sort_idx],
        y=mono_y[sort_idx],
        mode='lines',
        name='Monochromatic',
        line=dict(
            width=2,
            color='rgba(204, 121, 167, 1)'
        )
    ))
    fig.update_xaxes(
        title_text='Relative Polarization Angle (deg)',
        title_font=dict(size=20),
        showgrid=True,
        tickfont=dict(size=16),
        tickmode='array',
        tickvals=[0, 45, 90, 135, 180],
        range=[0, 180]
    )
    fig.update_yaxes(
        title_text='Ellipticity (-)',
        title_font=dict(size=20),
        showgrid=True,
        tickfont=dict(size=16),
        tickmode='array',
        tickvals=[0, 0.05, 0.1, 0.15, 0.2],
        range=[0, 0.20]
    )
    fig.update_layout(
        width=500,
        height=400,
        margin=dict(l=70, r=50, t=50, b=70),
        template='simple_white',
        font_family='crm12',
        legend=dict(
            font=dict(size=16),
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    fig.show()
    fig.write_image(r'mono_poly_plot.pdf', width=500, height=400)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        x_title='QWP Motor Angle (deg)',
        y_title='HWP Motor Angle (deg)',
        subplot_titles=('Monochromatic', 'Polychromatic')
    )
    fig.add_trace(go.Heatmap(
        z=monochromatic_ellipticity_map,
        x=qwp_angles,
        y=hwp_angles,
        coloraxis='coloraxis'
    ), row=1, col=1)
    fig.add_trace(go.Heatmap(
        z=polychromatic_ellipticity_map,
        x=qwp_angles,
        y=hwp_angles,
        coloraxis='coloraxis'
    ), row=2, col=1)
    # fig.add_trace(go.Scatter(
    #     x=qwp_angles[min_mono_indices+min_search_start],
    #     y=hwp_angles,
    #     opacity=0.4,
    #     mode='lines',
    #     line=dict(
    #         color='black',
    #         width=1.5
    #     ),
    #     showlegend=False
    # ), row=1, col=1)
    # fig.add_trace(go.Scatter(
    #     x=qwp_angles[min_poly_indices+min_search_start],
    #     y=hwp_angles,
    #     opacity=0.4,
    #     mode='lines',
    #     line=dict(
    #         color='black',
    #         width=1.5
    #     ),
    #     showlegend=False
    # ), row=2, col=1)
    fig.update_xaxes(
        tickmode='array',
        tickvals=[0, 30, 60, 90, 120, 150, 180],
        row=1, col=1
    )
    fig.update_xaxes(
        range=[0, 180],
        tickfont=dict(size=16),
        tickmode='array',
        tickvals=[0, 30, 60, 90, 120, 150, 180],
        row=2, col=1
    )
    fig.update_yaxes(
        range=[0, 90],
        tickfont=dict(size=16),
        tickmode='array',
        tickvals=[0, 30, 60, 90],
        row=1, col=1
    )
    fig.update_yaxes(
        range=[0, 90],
        tickfont=dict(size=16),
        tickmode='array',
        tickvals=[0, 30, 60, 90],
        row=2, col=1
    )
    fig.update_layout(
        width=500,
        height=400,
        margin=dict(l=70, r=50, t=50, b=70),
        template='simple_white',
        font_family='crm12',
        coloraxis=dict(
            cmin=0,
            cmax=1,
            colorscale=CUSTOM_COLORSCALE,
            colorbar_lenmode='pixels',
            colorbar_len=280,
            colorbar_thickness=15,
            colorbar_title='Ellipticity (-)',
            colorbar_title_font=dict(size=20),
            colorbar_tickfont=dict(size=16),
            colorbar_tickmode='array',
            colorbar_tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1],
            colorbar_ticktext=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'],
        ),
        annotations=[
            dict(
                font=dict(size=20)
            ) for annotation in fig.layout.annotations
        ]
    )
    fig.show()
    fig.write_image(r'mono_poly_map.pdf', width=500, height=400)

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
        [0.0, COLORS[5]+' 255)'],
        [0.5, 'rgba(255,255,255, 255)'],
        [1.0, COLORS[4]+' 255)']
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
            coloraxis='coloraxis'
        ))
        fig.add_trace(go.Scatter(
            x=np.rad2deg(phi_motor_solution_1),
            y=np.rad2deg(theta_motor),
            mode='markers',
            marker=dict(color='black', width=1)
        ))
        fig.add_trace(go.Scatter(
            x=np.rad2deg(phi_motor_solution_2),
            y=np.rad2deg(theta_motor),
            mode='markers',
            marker=dict(color='black', width=1)
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
        mode='markers',
        marker=dict(
            size=10,
            symbol='circle-dot',
            color='rgba(0, 0, 0, 0)',
            line=dict(
                width=3,
                color=COLORS[6]+', 255)'
            )
        )
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=fine_wl,
        y=fine_weights*np.amax(weights),
        mode='lines',
        line=dict(
            width=1.5,
            dash='dash',
            color=COLORS[6]+', 255)'
        )
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=wl_in_nm,
        y=retardance,
        mode='lines+markers',
        line=dict(
            width=1.5,
            dash='dash',
            color=COLORS[2]+', 255)'
        ),
        marker=dict(
            size=10,
            symbol='circle-dot',
            color='rgba(0, 0, 0, 0)',
            line=dict(
                width=3,
                color=COLORS[2]+', 255)'
            )
        )
    ), row=1, col=2)
    fig.update_xaxes(
        title_text='Wavelength (nm)',
        title_font=dict(size=20),
        showgrid=True,
        tickfont=dict(size=16),
        range=[860, 900],
        tickmode='array',
        tickvals=[860, 870, 880, 890, 900]
    )
    fig.update_yaxes(
        title_text='Relative Weight (-)',
        title_font=dict(size=20),
        showgrid=True,
        range=[0, None],
        tickfont=dict(size=16),
        row=1, col=1
    )
    fig.update_yaxes(
        title_text='Retardance (deg)',
        title_font=dict(size=20),
        tickfont=dict(size=16),
        showgrid=True,
        row=1, col=2
    )
    fig.update_layout(
        width=1000,
        height=400,
        margin=dict(l=70, r=50, t=30, b=90),
        template='simple_white',
        font_family='crm12',
        showlegend=False
    )
    fig.show()
    fig.write_image(r'bandwidth_and_spectral_dependance.pdf', width=1000, height=400)

def polarimeter_intensity(alpha: float, alpha_max: float, k: float, e_min: float) -> float:
    e_max = e_min + k**2
    return e_max**2 * cos(alpha_max - alpha)**2 + e_min**2 * sin(alpha_max - alpha)**2

def compute_polarization_parameters(angles: np.ndarray, intensity: np.ndarray, fit_factor: float = 1E4, max_intensity: float = 10):
    scaled_intensity = intensity * fit_factor
    max_scaled_intensity = max_intensity * fit_factor

    try:
        popt, _ = curve_fit(
            polarimeter_intensity, 
            angles, 
            scaled_intensity, 
            bounds=((0, 0, 0), (np.pi, max_scaled_intensity, max_scaled_intensity))
        )
    except RuntimeError:
        return -1, -1, None, np.inf

    alpha_max, k, e_min = popt

    e_max = k**2 + e_min
    ellipticity = e_min / e_max

    fitted_intensity = polarimeter_intensity(angles, *popt) / fit_factor

    rmse = np.sqrt(np.mean((intensity - fitted_intensity) ** 2))
    nrmse = rmse / np.mean(intensity)

    return ellipticity, e_max, alpha_max, fitted_intensity, nrmse

def compute_system_parameters(primes, aggregated_intensities):
    popt, _, _, msg, _ = curve_fit(
        general_intensity,
        primes,
        aggregated_intensities,
        p0 = [1, 1, 0, 0, 0, 0],
        bounds=([0, 0, -np.pi, -np.pi, -np.pi, -np.pi], [np.inf, np.inf, np.pi, np.pi, np.pi, np.pi]),
        full_output=True
    )

    fit = general_intensity(primes, *popt)
    rmse = np.sqrt(np.mean((aggregated_intensities - fit) ** 2))

    # print(popt, rmse)
    # print(msg)

    print(f"Intensity_0: {popt[0]:.2f}, Gamma: {popt[1]:.2f}, Delta: {np.rad2deg(popt[2]):.2f}, Theta_0: {np.rad2deg(popt[3]):.2f}, Phi_0: {np.rad2deg(popt[4]):.2f}, Alpha_0: {np.rad2deg(popt[5]):.2f}")

    # fig = go.Figure(data=go.Scatter(x=np.arange(len(aggregated_intensities)), y=aggregated_intensities))
    # fig.add_trace(go.Scatter(x=np.arange(len(aggregated_intensities)), y=fit))
    # fig.show()

    intensity_0 = popt[0]
    gamma = popt[1]
    delta = popt[2]
    theta_0 = popt[3]
    phi_0 = popt[4]
    alpha_0 = popt[5]

    return intensity_0, gamma, delta, theta_0, phi_0, alpha_0

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

def linear_polarization(phi, theta, delta):
    return np.tan(2*phi) + np.tan(delta) * np.sin(2 * (2*theta - phi))

def phi_motor_for_linear_polarization(theta_motor, theta_0, phi_0, delta, initial_guess=[0, 0]):
    theta = theta_motor - theta_0

    phi_motor_solution_1 = []
    phi_motor_solution_2 = []

    for t in theta:
        phi_solution_1 = root(linear_polarization, initial_guess[0], args=(t, delta), method='lm')['x'][0]
        phi_solution_2 = root(linear_polarization, initial_guess[1], args=(t, delta), method='lm')['x'][0]
        phi_motor_solution_1.append(phi_solution_1+phi_0)
        phi_motor_solution_2.append(phi_solution_2+phi_0)

    phi_motor_solution_1 = np.array(phi_motor_solution_1)
    phi_motor_solution_2 = np.array(phi_motor_solution_2)

    return phi_motor_solution_1, phi_motor_solution_2

if __name__ == '__main__':
    zos = zp.ZOS()
    oss = zos.connect('extension')
    oss.UpdateMode = zp.constants.LensUpdateMode.None_
    oss.TheApplication.ShowChangesInUI = False

    # monochromatic(oss)
    polychromatic(oss)
    # vary_dichroic_retardance(oss)
