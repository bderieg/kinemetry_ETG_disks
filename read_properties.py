import pandas as pd
import csv

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

def readp(filename):
    """ Reads a given properties file with each line of the format key=value.  Returns a dictionary containing the pairs.

    Keyword arguments:
        filename -- the name of the file to be read
    """
    result = {}
    with open(filename, "r") as csvfile:  # Open file as read-only
        # Define file-read object
        reader = csv.reader(csvfile, delimiter='=', escapechar='\\', quoting=csv.QUOTE_NONE)

        # Iterate through rows in file
        for row in reader:
            if len(row) == 0:  # If blank row
                continue
            elif len(row) != 2:  # If row doesn't make sense
                raise csv.Error("Parameter file syntax error on line "+str(row))
            try:  # Convert data types except for strings
                row[1] = eval(row[1])
            except SyntaxError:
                pass
            result[row[0].lower()] = row[1]  # Assign row to dictionary

    return result

try:
    properties = pd.read_excel(readp('config.param')['prop_filename'], engine='odf', sheet_name=readp('config.param')['sheet_name'], index_col=0)
except:
    properties = pd.read_csv(readp('config.param')['prop_filename'], index_col=0)

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
