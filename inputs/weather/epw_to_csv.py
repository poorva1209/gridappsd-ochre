import pandas as pd


def convert(file, results_fname):
	weather = pd.read_csv(file, skiprows=8,names=["Date","HH:MM","Datasource","DryBulb {C}","DewPoint {C}","RelHum {%}","Atmos Pressure {Pa}","ExtHorzRad {Wh/m2}","ExtDirRad {Wh/m2}","HorzIRSky {Wh/m2}","GloHorzRad {Wh/m2}","DirNormRad {Wh/m2}","DifHorzRad {Wh/m2}","GloHorzIllum {lux}","DirNormIllum {lux}",\
										"DifHorzIllum {lux}","ZenLum {Cd/m2}","WindDir {deg}","WindSpd {m/s}","TotSkyCvr {.1}","OpaqSkyCvr {.1}","Visibility {km}","Ceiling Hgt {m}","PresWeathObs","PresWeathCodes","Precip Wtr {mm}","Aerosol Opt Depth {.001}","SnowDepth {cm}","Days Last Snow","Albedo {.01}","Rain {mm}","Rain Quantity {hr}"])

	export = weather[["DryBulb {C}", "GloHorzRad {Wh/m2}"]]
	export.to_csv("hourly_dc.csv")
	export = export.append(pd.Series([0, 0], index=export.columns ), ignore_index=True)
	export.index = pd.date_range(start=0,periods=len(export),freq='H')#[:-1]
	# export.to_csv("original_weather_{}.csv".format(results_fname))
	export.resample('15min').pad()[:-1].to_csv(results_fname)
	# export = export.resample('15min').asfreq().interpolate()
	# export[:-1].to_csv(results_fname)




convert('USA_DC_Washington.National.724050_2018.epw', 'dc_weather_15min.csv')

