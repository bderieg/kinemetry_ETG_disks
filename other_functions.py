import csv
import matplotlib.pyplot as plt

################################
# Parameter file read function #
################################

def read_properties(filename):
    # Reads a given properties file with each line of the format key=value.  Returns a dictionary containing the pairs.
    # 
    # Keyword arguments:
    #     filename -- the name of the file to be read

    result = {}
    with open(filename, "r") as csvfile:  # Open file as read-only
        # Define file-read object
        reader = csv.reader(csvfile, delimiter='=', escapechar='\\', quoting=csv.QUOTE_NONE)

        # Iterate through rows in file
        for row in reader:
            row = [i.replace(" ","") for i in row]
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

##########################################
# Function to find centroid of a 2D list #
##########################################

def centroid(data):
    weighted_flux_x = 0.0
    weighted_flux_y = 0.0
    total_flux = 0.0

    for i,row in data.iterrows():
            weighted_flux_x += row['x (pix)']*row['mom0 (Jy/pix km/s)']
            weighted_flux_y += row['y (pix)']*row['mom0 (Jy/pix km/s)']
            total_flux += row['mom0 (Jy/pix km/s)']

    xc = weighted_flux_x/total_flux
    yc = weighted_flux_y/total_flux

    return xc, yc
