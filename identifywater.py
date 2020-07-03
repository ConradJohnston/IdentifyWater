"""
Read a QuantumESPRESSO scf output and identify which oxygen atoms are hydrated as OH or H2O,
based on a distance metric which correctly respects periodic triclinic cells. 
"""
import argparse
import numpy as np 

parser = argparse.ArgumentParser(description='Identify water and hydroxide molecules from a QE output file.')
parser.add_argument('filename', 
                    help='QE SCF output file to read')
parser.add_argument('--threshold', type=float, default=1.2,
                    help='Bond detection threshold in angstrom')

args = parser.parse_args()

BOHR_TO_ANG = 0.529177249

## Parse key data from QuantumESPRESSO output file
with open(args.filename, 'r') as fhandle:
    all_lines = fhandle.readlines()
    
cell_lines = []
coords = []
elements = []

for index, line in enumerate(all_lines):
    
    # Parse lattice parameter:
    if 'lattice parameter (alat)' in line:
        alat = float(line.split()[-2])*BOHR_TO_ANG

    # Parse number of atoms
    if 'number of atoms/cell' in line:
        n_atoms = int(line.split()[-1])
    
    # Parse cell
    if 'crystal axes' in line:
        for ii in range(3):
            cell_lines.append(all_lines[index+1+ii].split()[-4:-1])
    
    # Parse coordinates
    if 'Cartesian axes' in line:
        for ii in range(n_atoms):
            atom_line = all_lines[index+3+ii]
            elements.append(atom_line.split()[1])
            coords.append(atom_line.split()[-4:-1])
                    
        break

# Check where the first oxygen and hydrogen are - QE does some grouping and reordering by element
for index, elem in enumerate(elements):
    if elem is 'O':
        first_oxygen = index-1
        break
for index, elem in enumerate(elements):
    if elem is 'H':
        first_hydrogen = index-1
        break

cell = np.array(cell_lines, dtype=float).T*alat
coords_cartesian = np.array(coords,dtype=float)*alat
# Convert coords to crystal coordinates
coords_crystal = np.dot(coords_cartesian, np.linalg.inv(cell).T)

hydrated_oxygens = []

# Loop over oxygens
print("Finding bonds...")
for index_oxy, atom_type in enumerate(elements):
    if atom_type is 'O':
        # Loop over hydrogens
        bonded_hydrogens = []
        for index_hyd, atom_type in enumerate(elements):
            if atom_type is 'H':
                # Compute distance
                dist_vector = coords_crystal[index_hyd]-coords_crystal[index_oxy]
                dist_vector_min = dist_vector-np.rint(dist_vector) # Min image convention
                dist_vector_cartesian = np.dot(dist_vector_min, cell.T)
                dist = np.linalg.norm(dist_vector_cartesian)
                # Compare to threshold
                if dist < args.threshold:
                    print("O-{:<5} and H-{:<5} : {:.5f} Å".format(index_oxy-first_oxygen, index_hyd-first_hydrogen, dist))
                    bonded_hydrogens.append(index_hyd)                    
        if bonded_hydrogens:
            hydrated_oxygens.append((index_oxy, bonded_hydrogens))
        
# Categorise hydrated oxygens as hydroxides, waters, or flag as anomalies
hydroxides, waters = [], []
for tup in hydrated_oxygens:
    if len(tup[1]) == 1:
        hydroxides.append(tup[0])
    elif len(tup[1]) == 2:
        waters.append(tup[0])
    else:
        print("For the oxygen with index {}, I found more than two close hydrogens!".format(tup[0]+1))
        
# Print results to stdout
print("Found {} OH, with oxygen indicies:".format(len(hydroxides)))
for elem in hydroxides:
    print('{:<6} '.format('O-'+str(elem-first_oxygen)), end='')
print("")
for elem in hydroxides:
    print('{:<6} '.format(str(elem+1)), end='') # QE indices 
print("")

print("Found {} H2O, with oxygen indicies:".format(len(waters)))
for elem in waters:
    print('{:<6} '.format('O-'+str(elem-first_oxygen)), end='')
print("")
for elem in waters:
    print('{:<6} '.format(str(elem+1)), end='') # QE indices 
print("")
