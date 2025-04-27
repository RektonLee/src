import os
import re
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader
from sklearn.model_selection import train_test_split
from Bio.PDB import PDBParser, Selection
import logging
import subprocess
import tempfile
import requests
import xml.etree.ElementTree as ET
def extract_active_sites(xml_text):
    xml_text = xml_text.strip()  # 去掉前后的换行、空格
    root = ET.fromstring(xml_text)
    ns = {'u': 'http://uniprot.org/uniprot'}
    active_sites = []
    for feature in root.findall('.//u:feature[@type="active site"]', ns):
        pos_elem = feature.find('.//u:position', ns)
        if pos_elem is not None and 'position' in pos_elem.attrib:
            active_sites.append(int(pos_elem.attrib['position']))
    return active_sites
class ProteinStructureProcessor:
    def __init__(self, pdb_save_dir='output/pdb_files'):
        self.pdb_save_dir = pdb_save_dir
        os.makedirs(pdb_save_dir, exist_ok=True)
        self.uniprot_info_cache = {}

    def extract_binding_site(self, pdb_str, uniprot_id=None, sequence=None, radius=10.0):
        center = self.identify_binding_site(pdb_str, uniprot_id, sequence)
        with tempfile.NamedTemporaryFile('w', suffix=".pdb", delete=False) as tmp_file:
            tmp_file.write(pdb_str)
            tmp_pdb_path = tmp_file.name


        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', "tmp.pdb")
        model = structure[0]

        atoms = []
        atom_types = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'H': 5}
        residue_types = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        aa_3_to_1 = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }

        for residue in Selection.unfold_entities(model, 'R'):
            if residue.get_id()[0] != ' ':
                continue
            for atom in residue:
                coord = atom.get_coord()
                dist = np.linalg.norm(coord - center)
                if dist <= radius:
                    element = atom.element.strip() or atom.name[0]
                    atom_type = atom_types.get(element, 6)
                    aa1 = aa_3_to_1.get(residue.get_resname(), 'Y')
                    residue_index = residue_types.get(aa1, 19)

                    atom_type_one_hot = np.eye(6)[atom_type]
                    residue_one_hot = np.eye(20)[residue_index]
                    feature = np.concatenate([atom_type_one_hot, coord, [dist], residue_one_hot])
                    atoms.append((coord, feature))

        if os.path.exists(tmp_pdb_path):
            os.remove(tmp_pdb_path)

        coords, features = zip(*atoms)
        x = torch.tensor(np.array(features), dtype=torch.float)
        coords = np.array(coords)

        edge_index, edge_attr = [], []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < 5.0:
                    edge_index.extend([[i, j], [j, i]])
                    edge_attr.extend([[dist], [dist]])

        edge_index = torch.tensor(edge_index).T
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        return center, Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def identify_binding_site(self, structure_data, uniprot_id=None, sequence=None):
        with tempfile.NamedTemporaryFile('w', suffix='.pdb', delete=False) as tmp:
            tmp.write(structure_data)
            tmp_pdb_path = tmp.name

        try:
            if uniprot_id:
                active_site = self._get_active_site_from_uniprot(uniprot_id)
                if active_site:
                    logging.info(f"从UniProt获取到活性位点信息: {active_site}")
                    return self._get_coordinates_for_residues(tmp_pdb_path, active_site)

            pocket_centers = self._detect_pockets_with_fpocket(tmp_pdb_path)
            if pocket_centers is not None:
                logging.info(f"fpocket检测到口袋，使用得分最高的中心")
                return pocket_centers

            logging.warning("未识别到活性位点，使用原子坐标平均")
            return self._fallback_center(tmp_pdb_path)

        finally:
            if os.path.exists(tmp_pdb_path):
                os.remove(tmp_pdb_path)
            out_dir = tmp_pdb_path.replace('.pdb', '_out')
            if os.path.exists(out_dir):
                import shutil
                shutil.rmtree(out_dir)

    def _get_active_site_from_uniprot(self, uniprot_id):
        try:
            if uniprot_id not in self.uniprot_info_cache:
                url = f"https://www.uniprot.org/uniprot/{uniprot_id}.xml"
                response = requests.get(url)
                if response.status_code != 200:
                    return None
                self.uniprot_info_cache[uniprot_id] = response.text

            xml_text = self.uniprot_info_cache[uniprot_id]
            # pos = re.findall(r'<feature type="active site".+?position position="(\\d+)"', xml_text)
            # ranges = re.findall(r'<feature type="active site".+?begin position="(\\d+)".+?end position="(\\d+)"', xml_text)

            # result = [int(p) for p in pos]
            # for begin, end in ranges:
            #     result.extend(range(int(begin), int(end) + 1))
            active_sites = extract_active_sites(xml_text)
            return active_sites if active_sites else None
        except Exception as e:
            logging.error(f"获取UniProt活性位点失败: {e}")
            return None

    def _get_coordinates_for_residues(self, pdb_path, residue_indices):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        coords = []
        for residue in structure[0].get_residues():
            res_id = residue.get_id()[1]
            if res_id in residue_indices:
                for atom in residue:
                    coords.append(atom.get_coord())
        return np.mean(coords, axis=0)

    def _detect_pockets_with_fpocket(self, pdb_path):
        cmd = ["fpocket", "-f", pdb_path]
        output_dir = pdb_path.replace('.pdb', '_out')
        pocket_file = os.path.join(output_dir, "pockets", "pocket1_atm.pdb")

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if not os.path.exists(pocket_file):
                return None

            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("pocket", pocket_file)

            # 提取第一个原子的坐标
            for atom in structure.get_atoms():
                return atom.coord  # type: np.ndarray
            return None

        except Exception as e:
            logging.warning(f"fpocket运行或解析pocket1失败: {e}")
            return None

    def _fallback_center(self, pdb_path):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        coords = [atom.get_coord() for residue in structure[0].get_residues() for atom in residue]
        return np.mean(coords, axis=0) if coords else np.array([0, 0, 0])

def create_data_loaders(graphs, substrate_embeddings, y, train_idx, test_idx, batch_size=16):
    x_train = torch.tensor(substrate_embeddings[train_idx], dtype=torch.float32)
    x_test = torch.tensor(substrate_embeddings[test_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32)
    y_test = torch.tensor(y[test_idx], dtype=torch.float32)

    train_data = list(zip([graphs[i] for i in train_idx], x_train, y_train))
    test_data = list(zip([graphs[i] for i in test_idx], x_test, y_test))

    return GeometricDataLoader(train_data, batch_size=batch_size, shuffle=True), \
           GeometricDataLoader(test_data, batch_size=batch_size, shuffle=False)

def load_data_from_local(csv_path, npz_path, pdb_dir, max_samples=None):
    df = pd.read_csv(csv_path)
    if max_samples:
        df = df.iloc[:max_samples]
    npz_data = np.load(npz_path)
    processor = ProteinStructureProcessor(pdb_dir)

    graphs, substrates, targets, valid_idx = [], [], [], []
    graph_cache = {}  # 缓存 uniprot 对应的 graph

    for i, row in df.iterrows():
        uniprot = row['uniprot']
        pdb_path = os.path.join(pdb_dir, f"{uniprot}.pdb")
        if not os.path.exists(pdb_path):
            continue
        try:
            if uniprot not in graph_cache:
                with open(pdb_path) as f:
                    pdb_str = f.read()
                _, graph = processor.extract_binding_site(pdb_str, uniprot)
                graph_cache[uniprot] = graph
            else:
                graph = graph_cache[uniprot]
                logging.info(f"复用缓存的图结构: {uniprot}")

            graphs.append(graph)
            substrates.append(npz_data['substrate_embeddings'][i])
            targets.append([np.log10(row['Km Wildtype']), np.log10(row['kcat Wildtype'])])
            valid_idx.append(i)
        except Exception as e:
            logging.warning(f"处理 {uniprot} 失败: {e}")
            continue

    substrates = np.array(substrates)
    targets = np.array(targets)
    train_idx, test_idx = train_test_split(np.arange(len(graphs)), test_size=0.2, random_state=42)
    return create_data_loaders(graphs, substrates, targets, train_idx, test_idx)
