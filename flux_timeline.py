# flux_timeline.py

##########################
# import python packages #
##########################
import pandas as pd
import math
import tqdm


# Calculate magma area in m2 at each depth (x), set to 0 if shallower than top of the magma. Qm is the maximum magma area and la is the distance decay constant.
def Qd(x, Qm, la):
    if x < 0:
        value = 0.0
    else:
        value = Qm * math.exp(-1 * la * x)
    return value


# Maximum magma area in m2 assuming magma volume = Qmax(exp(-lambda*distance))
def Qmax_length(total_magma, duration_eruption, speed_magma):
    return (-(math.log(10.0**-3)) / 0.999) * (
        total_magma / (duration_eruption * speed_magma)
    )


# Lambda distance decay constant assuming magma volume = Qmax(exp(-lambda*distance))
def lambda_length(duration_eruption, speed_magma):
    return -(math.log(10.0**-3)) / (duration_eruption * speed_magma)


# calculate CO2 and S released at each depth given a magma column distribution
def magma_depth(type, degas_path, x, step_distance, Qm, la, constants):
    # depth of the top of the magma in m
    if type == "preeruption":
        # ascending one pressure step each loop before it reaches the surface
        top_magma = float((degas_path.loc[x, ["depth (m)"]]).iloc[0])
    elif type == "eruption":
        # once it reaches the surface (i.e., erupting), top of the magma column is in the air, but enables the calculation of the magma still underground
        top_magma = float(
            (degas_path.loc[(len(degas_path) - 1), ["depth (m)"]]).iloc[0]
        ) - (x * step_distance)
    # calculate depths relative to top of the magma in m
    degas_path["rel_magma_depth_m"] = degas_path["depth (m)"] - top_magma
    # calculate magma area in m2 at each depth, set to 0 if shallow than top of the magma
    degas_path["area_magma (m2)"] = degas_path.apply(
        lambda y: Qd(y["rel_magma_depth_m"], Qm, la), axis=1
    )
    # convert magma area to magma mass in kg at each depth
    degas_path["magma_mass_kg"] = (
        degas_path["area_magma (m2)"] * constants["density_magma"] * step_distance
    )
    # calculate CO2 released in kg at each depth
    degas_path["CO2_mass_kg"] = degas_path["wCO2g"] * degas_path["magma_mass_kg"]
    # calculate ST released in kg at each depth
    degas_path["ST_mass_kg"] = degas_path["wSTg"] * degas_path["magma_mass_kg"]
    return degas_path


# calculate flux during open-system degassing during magma ascent and eruption assuming magma volume is distributed with depth assuming Wadge curve for eruption and constant magma ascent rate
def calc_open_flux1(
    step_time, step_distance, time_, degas_path, Qm, la, header, end, type, constants
):
    # create a progress bar to display
    with tqdm.tqdm(total=(end)) as tqdmsteps:
        # x is the index on the degas_path file - essentially related to the depth top of the magma column
        for x in range(0, end, 1):
            # required for output of data
            y = x
            # need to add one if the mode is eruption otherwise it will repeat the last pre-eruption step
            if type == "eruption":
                x = x + 1
            # time relative to start of magma degassing in s
            time = time_ + x * step_time
            # calculate depth distribution
            degas_path = magma_depth(
                type, degas_path, x, step_distance, Qm, la, constants
            )
            # calculate total CO2 released in kg at this time
            CO2_total = degas_path["CO2_mass_kg"].sum()
            # calculate total CO2 released in kg at this time
            ST_total = degas_path["ST_mass_kg"].sum()
            # create results table
            result1 = pd.DataFrame([[time, CO2_total, ST_total]])
            if y == 0:
                # combine header and results
                result = pd.concat([header, result1])
            else:
                # add new results
                result = pd.concat([result, result1])
            # update progress bar
            tqdmsteps.update(1)
    # tidy up results table
    result.columns = result.iloc[0]
    result = result[1:]
    result.reset_index(drop=True, inplace=True)
    return result


# converts outputs of degassing calculation so that it can be used by the flux timeline function
def convert_degassing_calc(degas_path, tool, constants):
    # appropriate for EVo outputs
    if tool == "EVo":
        # convert pressure in bars to depth in m using crustal gradient in bar/km
        degas_path["depth (m)"] = (
            constants["crustal_gradient"] * degas_path["P (bars)"] * 1000.0
        )
        # calculate gas weight fraction at each step from gas wt% remaining at each step
        degas_path["gas_wt_loss"] = (
            (degas_path["Gas_wt"] / 100.0) / (1.0 - constants["gas_loss_fraction"])
        ) * constants["gas_loss_fraction"]
        # calculate weight fraction of total magma of CO2 and ST released at each depth
        degas_path["wCO2g"] = degas_path["gas_wt_loss"] * (
            degas_path["wCO2"]
            + (
                44.0095
                * ((degas_path["wCO"] / 28.0101) + (degas_path["wCH4"] / 16.0425))
            )
        )
        degas_path["wSTg"] = degas_path["gas_wt_loss"] * (
            degas_path["wS2"]
            + (
                32.066
                * ((degas_path["wSO2"] / 64.0638) + (degas_path["wH2S"] / 34.08088))
            )
        )
    return degas_path


# calculates the flux of CO2 and ST during open-system degassing
def calc_open_flux(scenario, degas_path, scenarios, constants, tool):

    # convert pressure to depth and calculate weight fraction total magma of CO2 and S released
    degas_path = convert_degassing_calc(degas_path, tool, constants)

    # eruption duration in s
    time_eruption = (
        (float((scenarios.loc["Eruption Duration (days)", [scenario]]).iloc[0]))
        * 24.0
        * 60.0
        * 60
    )
    # unrest duration in s
    time_unrest = (
        (float((scenarios.loc["detectable unrest (days)", [scenario]]).iloc[0]))
        * 24.0
        * 60.0
        * 60
    )  # s
    # magma speed in m/s assuming constant magma ascent
    speed_magma = (constants["depth_crust"] * 1000.0) / time_unrest
    # total erupted magma in m3
    total_magma = (
        float((scenarios.loc["DRE Magma Volume (m3)", [scenario]]).iloc[0])
        * constants["fraction_melt"]
    )

    # maximum area of magma
    Qm = Qmax_length(total_magma, time_eruption, speed_magma)
    # distance decay constant
    la = lambda_length(time_eruption, speed_magma)

    # distance between degassing calculation steps in m
    step_distance = constants["crustal_gradient"] * 1000.0 * constants["dP_step"]
    # time between degassing calculation steps in s
    step_time = step_distance / speed_magma

    # header for results table
    header = pd.DataFrame([["time (s)", "mass CO2 (kg)", "mass ST (kg)"]])

    # number of steps in degassing calculation
    end_preeruption = len(degas_path)
    end_eruption = int((time_eruption * speed_magma) / step_distance)

    # calculation for pre-eruptive and eruptive flux
    print("Part 1 of 2: Calculating pre-eruption flux")
    results_preeruption = calc_open_flux1(
        step_time,
        step_distance,
        0.0,
        degas_path,
        Qm,
        la,
        header,
        end_preeruption,
        "preeruption",
        constants,
    )
    print("Part 2 of 2: Calculating eruptive flux")
    results_eruption = calc_open_flux1(
        step_time,
        step_distance,
        step_time * end_preeruption,
        degas_path,
        Qm,
        la,
        header,
        end_eruption,
        "eruption",
        constants,
    )

    # tidy up results table
    result = pd.concat([results_preeruption, results_eruption])
    result.reset_index(drop=True, inplace=True)

    # convert time to days relative to start of eruption
    time_to_eruption_days = (
        float((degas_path.loc[0, ["depth (m)"]]).iloc[0]) / speed_magma
    ) / (24.0 * 60.0 * 60)
    result["time wrt eruption start (days)"] = (
        result["time (s)"] / (24.0 * 60.0 * 60)
    ) - time_to_eruption_days

    # convert to ton/day
    result["CO2 flux (ton/day)"] = (result["mass CO2 (kg)"] / 1000.0) / (
        step_time / (24.0 * 60.0 * 60.0)
    )
    result["ST flux (ton/day)"] = (result["mass ST (kg)"] / 1000.0) / (
        step_time / (24.0 * 60.0 * 60.0)
    )

    return result


# calculate total emissions for a given magma distribution
def total_emissions(result):
    CO2 = (result["mass CO2 (kg)"].sum()) / 1.0e9
    ST = (result["mass ST (kg)"].sum()) / 1.0e9
    return CO2, ST


# calculate open-system flux per day
def calc_open_flux_days(flux, scenarios, scenario):

    header = pd.DataFrame(
        [["time wrt eruption start (days)", "mass CO2 (kg)", "mass ST (kg)"]]
    )

    days_eruption = float(
        (scenarios.loc["Eruption Duration (days)", [scenario]]).iloc[0]
    )
    days_unrest = float((scenarios.loc["detectable unrest (days)", [scenario]]).iloc[0])

    for days in range(int(-1 * days_unrest), int(days_eruption), 1):
        one_day = flux[flux["time wrt eruption start (days)"] > days]
        one_day = one_day[one_day["time wrt eruption start (days)"] < (days + 1)]
        CO2_total = one_day["mass CO2 (kg)"].sum()
        ST_total = one_day["mass ST (kg)"].sum()
        # create results table
        result1 = pd.DataFrame([[days, CO2_total, ST_total]])
        if days == (-1 * days_unrest):
            # combine header and results
            result = pd.concat([header, result1])
        else:
            # add new results
            result = pd.concat([result, result1])

    result.columns = result.iloc[0]
    result = result[1:]
    result.reset_index(drop=True, inplace=True)

    return result


# calculate closed-system flux using Wadge curve
def calc_closed_flux(scenario, scenarios):
    total_CO2 = float((scenarios.loc["CO2 (Mt)", [scenario]]).iloc[0])  # Mt
    total_ST = float((scenarios.loc["ST (Mt)", [scenario]]).iloc[0])  # Mt
    t_days = float(
        (scenarios.loc["Eruption Duration (days)", [scenario]]).iloc[0]
    )  # days
    t_hr = int(t_days * 24.0)  # hour
    la = (3.0 * math.log(10.0)) / t_days  # lambda in days
    Qm_CO2 = ((3.0 * math.log(10.0)) * total_CO2 * 1000000) / (
        0.999 * t_days
    )  # ton/day
    Qm_ST = ((3.0 * math.log(10.0)) * total_ST * 1000000) / (0.999 * t_days)  # ton/day
    header = pd.DataFrame(
        [
            [
                "time (days)",
                "mass CO2 (kg)",
                "mass ST (kg)",
                "CO2 flux (ton/day)",
                "ST flux (ton/day)",
            ]
        ]
    )
    for t in range(0, t_hr, 1):
        time = t / (24.0)
        Q_CO2 = Qm_CO2 * math.exp(-1.0 * la * time)  # ton/day
        Q_ST = Qm_ST * math.exp(-1.0 * la * time)  # ton/day
        M_CO2 = (Q_CO2 / (24.0)) * 1e3  # kg
        M_ST = (Q_ST / (24.0)) * 1e3  # kg
        result1 = pd.DataFrame([[time, M_CO2, M_ST, Q_CO2, Q_ST]])
        if t == 0:
            # combine header and results
            result = pd.concat([header, result1])
        else:
            # add new results
            result = pd.concat([result, result1])
    # tidy up results table
    result.columns = result.iloc[0]
    result = result[1:]
    result.reset_index(drop=True, inplace=True)
    print(Qm_CO2, Qm_ST)
    return result


# calculate depth distribution of magma
def depth_curves(x, type, scenario, degas_path, scenarios, constants, tool):

    # convert pressure to depth and calculate weight fraction total magma of CO2 and S released
    degas_path = convert_degassing_calc(degas_path, tool, constants)

    # eruption duration in s
    time_eruption = (
        (float((scenarios.loc["Eruption Duration (days)", [scenario]]).iloc[0]))
        * 24.0
        * 60.0
        * 60
    )
    # unrest duration in s
    time_unrest = (
        (float((scenarios.loc["detectable unrest (days)", [scenario]]).iloc[0]))
        * 24.0
        * 60.0
        * 60
    )  # s
    # magma speed in m/s assuming constant magma ascent
    speed_magma = (constants["depth_crust"] * 1000.0) / time_unrest
    # total erupted magma in m3
    total_magma = (
        float((scenarios.loc["DRE Magma Volume (m3)", [scenario]]).iloc[0])
        * constants["fraction_melt"]
    )

    # maximum area of magma
    Qm = Qmax_length(total_magma, time_eruption, speed_magma)
    # distance decay constant
    la = lambda_length(time_eruption, speed_magma)

    # distance between degassing calculation steps in m
    step_distance = constants["crustal_gradient"] * 1000.0 * constants["dP_step"]

    degas_path = magma_depth(type, degas_path, x, step_distance, Qm, la, constants)

    return degas_path
