from dwelling_model import Psychrometrics_HVAC

PsyCalc = Psychrometrics_HVAC.Psychrometrics()

h_vap = 2454  # kJ/kg
humidity_cap_mult = 15.0


class HumidityModel:
    def __init__(self, t_indoor, time_res, **kwargs):
        """
        Dwelling humidity model
        """
        self.time_res = time_res

        # Ventilated space parameters
        self.indoor_volume = kwargs['building volume (m^3)']

        # Initial conditions
        p_outdoor = kwargs['initial_schedule']['ambient_pressure']
        # self.indoor_rh = kwargs['initial_schedule']['ambient_humidity']
        self.indoor_rh = 0.25  # TODO: Forcing RH to 25% for now
        self.indoor_w = PsyCalc.w_fT_R_P_SI(t_indoor, self.indoor_rh, p_outdoor)
        self.indoor_density = PsyCalc.rhoD_fT_w_P_SI(t_indoor, self.indoor_w, p_outdoor)  # (kg/m^3)
        self.indoor_wet_bulb = PsyCalc.Twb_fT_R_P_SI(t_indoor, self.indoor_rh, p_outdoor)

    def update_humidity(self, t_indoor, ach_indoor, t_outdoor, rh_outdoor, p_outdoor, latent_gains):
        """
        Update dwelling humidity given:
            Occupancy metabolism
            Appliance latent gains
            Wind speed (air changes)
            Outside pressure
            HVAC latent cooling

            Inputs are in units of degC, kPa, fraction (for RH), W (for latent gains)
        """
        # FUTURE: Dehumidifier? Latent portion of HPWH gains?

        latent_gains_w = latent_gains * self.time_res.total_seconds() / 1000 / (
                self.indoor_density * self.indoor_volume * h_vap)  # unitless latent gains
        # outdoor_w = Hum_rat_Tdb_RH_P_fnc(t_outdoor, rh_outdoor, p_outdoor)
        outdoor_w = PsyCalc.w_fT_R_P_SI(t_outdoor, rh_outdoor, p_outdoor)

        # Update moisture balance calculations
        hours = self.time_res.total_seconds() / 3600
        self.indoor_w += (latent_gains_w - ach_indoor * hours * (self.indoor_w - outdoor_w)) / humidity_cap_mult
        if self.indoor_w < 0:
            self.indoor_w = 0
            #TODO: comment this out while we're doing test suite runs (we have negative latent gains for the first season), but uncoment for real buildings
            #print("Warning: Indoor Relative Humidity less than 0%, double check inputs.")

        self.indoor_rh = PsyCalc.R_fT_w_P_SI(t_indoor, self.indoor_w, p_outdoor)
        if self.indoor_rh > 1:
            print("WARNING: Indoor Relative Humidity greater than 100%, condensation is occuring.")
            self.indoor_rh = 1
            self.indoor_w = PsyCalc.w_fT_R_P_SI(t_indoor, self.indoor_rh, p_outdoor)

        self.indoor_density = PsyCalc.rhoD_fT_w_P_SI(t_indoor, self.indoor_w, p_outdoor)  # kg/m^3

        self.indoor_wet_bulb = PsyCalc.Twb_fT_R_P_SI(t_indoor, self.indoor_rh, p_outdoor)
