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

properties = pd.read_csv('/home/ben/Desktop/research/research_boizelle_working/kinemetry/galaxy_properties.csv', index_col=0)

for key,data in properties.iterrows():
    if any([True if p in key else False for p in prefixes]):
        newname = key.split('(', 1)[0]  # Remove parentheses
        newname = newname.replace(' ','')  # Remove spaces
        properties.rename(index={key:newname}, inplace=True)

def get_prop(target, prop):
    return properties.loc[target][prop]
