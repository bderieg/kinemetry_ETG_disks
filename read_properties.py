import pandas as pd

# Define some constants
prefixes = [
        "ESO",
        "NGC",
        "FCC",
        "GAMA",
        "CGCG",
        "Hydra",
        "IC",
        "PGC",
        "PKS",
        "UGC",
        "MCG",
        "WISEA"
        ]

properties = pd.read_excel('/home/ben/Desktop/research/research_boizelle_working/kinemetry_working/kinemetry_progress.ods', engine='odf', sheet_name='Target Parameters', index_col=0)

for key,data in properties.iterrows():
    try:
        if any([True if p in key else False for p in prefixes]):
            newname = key.split('(', 1)[0]  # Remove parentheses
            newname = newname.replace(' ','')  # Remove spaces
            properties.rename(index={key:newname}, inplace=True)
    except TypeError:
        pass

def get_prop(target, prop):
    return properties.loc[target][prop]
