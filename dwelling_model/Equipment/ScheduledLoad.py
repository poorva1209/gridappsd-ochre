import os
import pandas as pd

from dwelling_model import default_input_path
from dwelling_model.Equipment import Equipment, EquipmentException
from dwelling_model.FileIO import import_generic


class ScheduledLoad(Equipment):
    """
    Equipment with a pre-defined schedule. Schedule may come from:

    - The main schedule and properties files. The schedule must have a column with the same name as the equipment that
     corresponds to the fraction of full load at a given time. The properties file must include the full load power in
     Watts, and gain fractions for convective, radiative, and latent gains.
    - A separate schedule file, named `equipment_schedule_file`. The schedule must be defined for a full year at
     anywhere between 1 minute and 1 hour resolution. It must have a column that specifies the power output in kW. Note
     that heat gains are not considered for this type of schedule.
    """

    def __init__(self, full_schedule=None, equipment_schedule=None, properties_name=None, **kwargs):
        super().__init__(**kwargs)

        self.power = 0  # in kW
        self.sensible_gain_fraction = 0  # unitless
        self.latent_gain_fraction = 0  # unitless

        if equipment_schedule is not None:
            # take schedule as is - used for PV; for now, no heat gains
            self.schedule = equipment_schedule
        elif full_schedule is not None and self.name in full_schedule and properties_name is not None:
            # load from main schedule and properties file - schedule is a fraction of max power
            self.schedule = full_schedule.loc[:, self.name].copy()  # assumes only 1 column, convert to pd.Series

            # Initialize from properties file
            # FUTURE: standardize the properties file for non-controllable loads
            if properties_name + ' power' in kwargs:
                kwargs[properties_name + ' power (W)'] = kwargs[properties_name + ' power']
            if properties_name + ' power elec' in kwargs:
                kwargs[properties_name + ' power (W)'] = kwargs[properties_name + ' power elec']
            if properties_name + ' electric power (W)' in kwargs:
                kwargs[properties_name + ' power (W)'] = kwargs[properties_name + ' electric power (W)']

            if properties_name + ' power (W)' not in kwargs:
                raise EquipmentException('Properties could not be parsed for {}.'.format(self.name))

            # Sensible and latent gain fractions, unitless
            self.sensible_gain_fraction = (kwargs[properties_name + ' convective gainfrac'] +
                                           kwargs[properties_name + ' radiative gainfrac'])
            self.latent_gain_fraction = kwargs[properties_name + ' latent gainfrac']

            max_power = kwargs[properties_name + ' power (W)']
            self.schedule *= max_power / 1000  # W to kW
        elif 'equipment_schedule_file' in kwargs:
            # load schedule from separate schedule file, for now, no heat gains
            self.schedule = self.import_schedule(**kwargs)
        else:
            raise EquipmentException('No schedule found for {}'.format(self.name))

        # check timing of schedule file
        schedule_res = self.schedule.index[1] - self.schedule.index[0]
        if self.time_res != schedule_res or self.start_time != self.schedule.index[0]:
            raise EquipmentException('Times in {} schedule differ from equipment arguments'.format(self.name))

        self.schedule_iterable = None
        self.reset_time()

    def import_schedule(self, equipment_schedule_file, val_col='Power', schedule_scale_factor=1,
                        input_path=default_input_path, **kwargs):
        if not os.path.isabs(equipment_schedule_file):
            equipment_schedule_file = os.path.join(input_path, self.name, equipment_schedule_file)
        schedule = import_generic(equipment_schedule_file, annual_input=True, keep_cols=[val_col], **kwargs)
        schedule = schedule[val_col] * schedule_scale_factor
        return schedule

    def reset_time(self):
        super().reset_time()
        if isinstance(self.schedule, pd.DataFrame):
            self.schedule_iterable = self.schedule.itertuples()
        else:
            self.schedule_iterable = iter(self.schedule)

    def update_external_control(self, schedule, ext_control_args):
        # Duty cycle allows power to reduce (or turn off), e.g. for resilience cases
        load_fraction = ext_control_args.get('Load Fraction', 1)

        self.update_internal_control(schedule)
        self.power *= load_fraction

        return 'On' if self.power > 0 else 'Off'

    def update_internal_control(self, schedule):
        self.power = next(self.schedule_iterable)

        return 'On' if self.power > 0 else 'Off'

    def calculate_power_and_heat(self, schedule):
        self.electric_kw = self.power  # kW
        self.sensible_gain = self.power * self.sensible_gain_fraction * 1000  # W
        self.latent_gain = self.power * self.latent_gain_fraction * 1000  # W

    def generate_results(self, verbosity, to_ext=False):
        results = super().generate_results(verbosity, to_ext)
        if not to_ext:
            if verbosity >= 6 and self.name != self.end_use:
                # assumes electric and not gas equipment
                results.update({self.name + ' Electric Power (kW)': self.electric_kw,
                                self.name + ' Reactive Power (kVAR)': self.reactive_kvar})
        return results
