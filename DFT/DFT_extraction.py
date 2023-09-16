import os
import re
import pandas as pd
import sys
sys.path.append("D:/Reaction optimization project/NiCOlit work reproduction/NiCOlit/DFT")

from autoqchem.descriptor_functions import *

logger = logging.getLogger(__name__)
float_or_int_regex = "[-+]?[0-9]*\.[0-9]+|[0-9]+"

class gaussian_log_extractor(object):
    """"""

    def __init__(self, log_file_path):
        """Initialize the log extractor. Extract molecule geometry and atom labels.

        :param log_file_path: local path of the log file
        :type log_file_path: str
        """

        with open(log_file_path) as f:
            self.log = f.read()
        self.log_file_path = log_file_path  # record keeping; used to output log file path in case of exception

        # initialize descriptors
        self.descriptors = {}
        self.atom_freq_descriptors = None
        self.atom_td_descriptors = None
        self.atom_descriptors = None
        self.vbur = None
        self.modes = None
        self.mode_vectors = None
        self.transitions = None
        self.n_tasks = len(re.findall("Normal termination", self.log))

        self._split_parts()  # split parts
        
    def get_geometry(self) ->  pd.DataFrame:
        """Extract geometry dataframe and store as attribute 'geom'."""

        # regex logic: find parts between "Standard orientation.*X Y Z" and "Rotational constants"
        geoms = re.findall("Standard orientation:.*?X\s+Y\s+Z\n(.*?)\n\s*Rotational constants",
                           self.log, re.DOTALL)

        # use the last available geometry block
        geom = geoms[-1]
        geom = map(str.strip, geom.splitlines())  # split lines and strip outer spaces
        geom = filter(lambda line: set(line) != {'-'}, geom)  # remove lines that only contain "---"
        geom = map(str.split, geom)  # split each line by space

        # convert to np.array for further manipulation (note: arrays have unique dtype, here it's str)
        geom_arr = np.array(list(geom))
        # create a dataframe
        geom_df = pd.concat([
            pd.DataFrame(geom_arr[:, 1:3].astype(int), columns=['AN', 'Type']),
            pd.DataFrame(geom_arr[:, 3:].astype(float), columns=list('XYZ'))
        ], axis=1)

        self.geom = geom_df    
        return geom_df
    def get_atom_labels(self) -> list:
        """Fetch the atom labels and store as attribute 'labels'."""

        # regex logic, fetch part between "Multiplicity =\d\n" and a double line
        # break (empty line may contain spaces)
        z_matrix = re.findall("Multiplicity = \d\n(.*?)\n\s*\n", self.log, re.DOTALL)[0]
        z_matrix = list(map(str.strip, z_matrix.split("\n")))
        # clean up extra lines if present
        if z_matrix[0].lower().startswith(('redundant', 'symbolic')):
            z_matrix = z_matrix[1:]
        if z_matrix[-1].lower().startswith('recover'):
            z_matrix = z_matrix[:-1]

        # fetch labels checking either space or comma split
        self.labels = []
        for line in z_matrix:
            space_split = line.split()
            comma_split = line.split(",")
            if len(space_split) > 1:
                self.labels.append(space_split[0])
            elif len(comma_split) > 1:
                self.labels.append(comma_split[0])
            else:
                raise Exception("Cannot fetch labels from geometry block")
        ls = []
        ls = self.labels 
        return ls
    
    def _split_parts(self) -> None:
        """Split the log file into parts that correspond to gaussian tasks."""

        # regex logic: log parts start with a new line and " # " pattern
        log_parts = re.split("\n\s-+\n\s#\s", self.log)[1:]
        self.parts = {}
        for p in log_parts:
            # regex logic: find first word in the text
            name = re.search("^\w+", p).group(0).lower()
            self.parts[name] = p
    
    def get_freq_part_descriptors(self) -> pd.DataFrame:
        """Extract descriptors from frequency part."""

        logger.debug("Extracting frequency section descriptors")
        if 'opt freq' not in self.parts:
            logger.info("Output file does not have a 'freq' section. Cannot extract descriptors.")
            #return

        text = self.log

        # single value descriptors
        single_value_desc_list = [
            {"name": "number_of_atoms", "prefix": "NAtoms=\s*", "type": int},
            {"name": "charge", "prefix": "Charge\s=\s*", "type": int},
            {"name": "multiplicity", "prefix": "Multiplicity\s=\s*", "type": int},
            {"name": "dipole", "prefix": "Dipole moment \(field-independent basis, Debye\):.*?Tot=\s*", "type": float},
            {"name": "molar_mass", "prefix": "Molar Mass =\s*", "type": float},
            {"name": "molar_volume", "prefix": "Molar volume =\s*", "type": float},
            {"name": "electronic_spatial_extent", "prefix": "Electronic spatial extent\s+\(au\):\s+<R\*\*2>=\s*",
             "type": float},
            {"name": "E_scf", "prefix": "SCF Done:\s+E.*?=\s*", "type": float},
            {"name": "zero_point_correction", "prefix": "Zero-point correction=\s*", "type": float},
            {"name": "E_thermal_correction", "prefix": "Thermal correction to Energy=\s*", "type": float},
            {"name": "H_thermal_correction", "prefix": "Thermal correction to Enthalpy=\s*", "type": float},
            {"name": "G_thermal_correction", "prefix": "Thermal correction to Gibbs Free Energy=\s*", "type": float},
            {"name": "E_zpe", "prefix": "Sum of electronic and zero-point Energies=\s*", "type": float},
            {"name": "E", "prefix": "Sum of electronic and thermal Energies=\s*", "type": float},
            {"name": "H", "prefix": "Sum of electronic and thermal Enthalpies=\s*", "type": float},
            {"name": "G", "prefix": "Sum of electronic and thermal Free Energies=\s*", "type": float},
        ]

        for desc in single_value_desc_list:
            for part_name in ['freq', 'opt']:
                try:
                    value = re.search(f"{desc['prefix']}({float_or_int_regex})",
                                      self.parts[part_name],
                                      re.DOTALL).group(1)
                    self.descriptors[desc["name"]] = desc['type'](value)
                except (AttributeError, KeyError):
                    pass
            if desc["name"] not in self.descriptors:
                self.descriptors[desc["name"]] = None
                logger.warning(f'''Descriptor {desc["name"]} not present in the log file.''')

        # stoichiometry
        self.descriptors['stoichiometry'] = re.search("Stoichiometry\s*(\w+)", text).group(1)

       

        # energies, regex-logic: find all floats in energy block, split by occupied, virtual orbitals
        string = re.search("Population.*?SCF [Dd]ensity.*?(\sAlph.*?)\n\s*Condensed", text, re.DOTALL).group(1)
        if self.descriptors['multiplicity'] == 1:
            energies = [re.findall(f"({float_or_int_regex})", s_part) for s_part in string.split("Alpha virt.", 1)]
            occupied_energies, unoccupied_energies = [map(float, e) for e in energies]
            homo, lumo = max(occupied_energies), min(unoccupied_energies)
        elif self.descriptors['multiplicity'] == 3:
            alpha, beta = re.search("(\s+Alpha\s+occ. .*?)(\s+Beta\s+occ. .*)", string, re.DOTALL).groups()
            energies_alpha = [re.findall(f"({float_or_int_regex})", s_part) for s_part in alpha.split("Alpha virt.", 1)]
            energies_beta = [re.findall(f"({float_or_int_regex})", s_part) for s_part in beta.split("Beta virt.", 1)]
            occupied_energies_alpha, unoccupied_energies_alpha = [map(float, e) for e in energies_alpha]
            occupied_energies_beta, unoccupied_energies_beta = [map(float, e) for e in energies_beta]
            homo_alpha, lumo_alpha = max(occupied_energies_alpha), min(unoccupied_energies_alpha)
            homo_beta, lumo_beta = max(occupied_energies_beta), min(unoccupied_energies_beta)
            homo, lumo = homo_alpha, lumo_beta
        else:
            logger.warning(f"Unsupported multiplicity {self.descriptors['multiplicity']}, cannot compute homo/lumo. "
                           f"Setting both to 0.")
            homo, lumo = 0, 0
        self.descriptors['homo_energy'] = homo
        self.descriptors['lumo_energy'] = lumo
        self.descriptors['electronegativity'] = -0.5 * (lumo + homo)
        self.descriptors['hardness'] = 0.5 * (lumo - homo)
        
        data = self.descriptors
        df = pd.DataFrame.from_dict(data, orient='index', columns=['Value']).T
        return df

        # atom_dependent section
        # Mulliken population
        string = re.search("Mulliken charges.*?\n(.*?)\n\s*Sum of Mulliken", text, re.DOTALL).group(1)
        charges = np.array(list(map(str.split, string.splitlines()))[1:])[:, 2]
        if len(charges) < len(self.labels):
            string = re.search("Mulliken atomic charges.*?\n(.*?)\n\s*Sum of Mulliken", text, re.DOTALL).group(1)
            charges = np.array(list(map(str.split, string.splitlines()))[1:])[:, 2]
        mulliken = pd.Series(charges, name='Mulliken_charge')

        # APT charges
        try:
            string = re.search("APT charges.*?\n(.*?)\n\s*Sum of APT", text, re.DOTALL).group(1)
            charges = np.array(list(map(str.split, string.splitlines()))[1:])[:, 2]
            apt = pd.Series(charges, name='APT_charge')
        except (IndexError, AttributeError):
            try:
                string = re.search("APT atomic charges.*?\n(.*?)\n\s*Sum of APT", text, re.DOTALL).group(1)
                charges = np.array(list(map(str.split, string.splitlines()))[1:])[:, 2]
                apt = pd.Series(charges, name='APT_charge')
            except Exception:
                apt = pd.Series(name='APT_charge')
                logger.warning(f"Log file does not contain APT charges.")

        # NPA charges
        try:
            string = re.search("Summary of Natural Population Analysis:.*?\n\s-+\n(.*?)\n\s=+\n", text,
                               re.DOTALL).group(1)
            population = np.array(list(map(str.split, string.splitlines())))[:, 2:]
            npa = pd.DataFrame(population,
                               columns=['NPA_charge', 'NPA_core', 'NPA_valence', 'NPA_Rydberg', 'NPA_total'])
        except Exception:
            npa = pd.DataFrame(['NPA_charge', 'NPA_core', 'NPA_valence', 'NPA_Rydberg', 'NPA_total'])
            logger.warning(f"Log file does not contain NPA charges.")

        # NMR
        try:
            string = re.findall(f"Isotropic\s=\s*({float_or_int_regex})\s*Anisotropy\s=\s*({float_or_int_regex})", text)
            nmr = pd.DataFrame(np.array(string).astype(float), columns=['NMR_shift', 'NMR_anisotropy'])
        except Exception:
            nmr = pd.DataFrame(columns=['NMR_shift', 'NMR_anisotropy'])
            logger.warning(f"Log file does not contain NMR shifts.")

        self.atom_freq_descriptors = pd.concat([mulliken, apt, npa, nmr], axis=1)

    def _get_td_part_descriptors(self) -> None:
        """Extract descriptors from TD part."""

        logger.debug("Extracting TD section descriptors")
        if 'td' not in self.parts:
            logger.info("Output file does not have a 'TD' section. Cannot extract descriptors.")
            return

        text = self.parts['td']

        single_value_desc_list = [
            {"name": "ES_root_dipole", "prefix": "Dipole moment \(field-.*?, Debye\):.*?Tot=\s*", "type": float},
            {"name": "ES_root_molar_volume", "prefix": "Molar volume =\s*", "type": float},
            {"name": "ES_root_electronic_spatial_extent",
             "prefix": "Electronic spatial extent\s+\(au\):\s+<R\*\*2>=\s*", "type": float},
        ]

        for desc in single_value_desc_list:
            value = re.search(f"{desc['prefix']}({float_or_int_regex})", text, re.DOTALL).group(1)
            self.descriptors[desc["name"]] = desc['type'](value)

        # excited states
        string = re.findall(f"Excited State.*?({float_or_int_regex})\snm"
                            f".*f=({float_or_int_regex})"
                            f".*<S\*\*2>=({float_or_int_regex})", text)
        self.transitions = pd.DataFrame(np.array(string).astype(float),
                                        columns=['ES_transition', 'ES_osc_strength', 'ES_<S**2>'])

        # atom_dependent section
        # Mulliken population
        string = re.search("Mulliken charges.*?\n(.*?)\n\s*Sum of Mulliken", text, re.DOTALL).group(1)
        charges = np.array(list(map(str.split, string.splitlines()))[1:])[:, 2]
        mulliken = pd.Series(charges, name='ES_root_Mulliken_charge')

        # NPA charges
        string = re.search("Summary of Natural Population Analysis:.*?\n\s-+\n(.*?)\n\s=+\n", text, re.DOTALL).group(1)
        population = np.array(list(map(str.split, string.splitlines())))[:, 2:]
        npa = pd.DataFrame(population, columns=['ES_root_NPA_charge', 'ES_root_NPA_core', 'ES_root_NPA_valence',
                                                'ES_root_NPA_Rydberg', 'ES_root_NPA_total'])

        self.atom_td_descriptors = pd.concat([mulliken, npa], axis=1)

"""
extract1 = gaussian_log_extractor("D:/Reaction optimization project/NiCOlit work reproduction/NiCOlit/DFT/testdata/LA.log")
#df1 = extract.get_geometry()
#ls = extract.get_atom_labels()
df1 = extract1.get_freq_part_descriptors()
#df1.to_csv("D:/Reaction optimization project/NiCOlit work reproduction/NiCOlit/DFT/testdatageom1.csv")

extract2 = gaussian_log_extractor("D:/Reaction optimization project/NiCOlit work reproduction/NiCOlit/DFT/testdata/LB.")
#df1 = extract.get_geometry()
#ls = extract.get_atom_labels()
df2 = extract2.get_freq_part_descriptors()
merged_df = pd.concat([df1, df2], ignore_index=True)
merged_df.to_csv("D:/Reaction optimization project/NiCOlit work reproduction/NiCOlit/DFT/testdatageom2.csv")

"""
def extract_descriptors_from_folder(folder_path):
    log_files = [file for file in os.listdir(folder_path) if file.endswith(".LOG") or file.endswith(".log") or file.endswith(".out")]
    merged_df = pd.DataFrame()

    for log_file in log_files:
        log_file_path = os.path.join(folder_path, log_file)
        log_extractor = gaussian_log_extractor(log_file_path)
        df = log_extractor.get_freq_part_descriptors()
        df.insert(0, 'FileName', log_file)
        merged_df = merged_df.append(df, ignore_index=True)
       
    return merged_df


df = extract_descriptors_from_folder("D:/Reaction optimization project/source code/DFT/opti/ligand/optimized structure")
df.to_csv("D:/Reaction optimization project/source code/DFT/opti/ligand/ligand_trial_descriptors.csv")