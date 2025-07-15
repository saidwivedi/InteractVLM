import base64
import os, json, cv2, torch, numpy as np
import os.path as op
import joblib as jl
from tqdm import tqdm
from smplx import build_layer
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

from render_mesh_utils import (
    compute_vertex_normals, render_mesh,
    project_vertices_and_create_mask
)
from render_mesh_utils import VIRTUVIAN_POSE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BODY_MODEL_PATH = 'data/body_models/smplh/SMPLH_NEUTRAL.pkl'
MERGED_SEGM = jl.load('./data/smpl_segmentation_merged.pkl')

TSV_IMG_FILE = '/ps/project/datasets/RICH/for_bstro_training/rich_for_bstro_tsv_db/train.img.tsv'
TSV_LABEL_FILE = '/ps/project/datasets/RICH/for_bstro_training/rich_for_bstro_tsv_db/train.label.tsv'
TSV_LINE_LIST = '/ps/project/datasets/RICH/for_bstro_training/rich_for_bstro_tsv_db/train.linelist.tsv'

OUTPUT_ROOT = '/is/cluster/fast/sdwivedi/work/lemon_3d/Data/rich'

RENDER_IMG_SIZE = (1024, 1024)
VIEWS = {
    'topfront': (2, 45, 315, 0., 0.0),
    'topback': (2, 45, 135, 0., 0.0),
    'bottomfront': (2, 315, 315, 0., 0.3),
    'bottomback': (2, 315, 135, 0., 0.3),
}

# Function and classes for reading TSV files and processing images taken from
# https://github.com/paulchhuang/bstro

def read_to_character(fp, c):
    result = []
    while True:
        s = fp.read(32)
        assert s != ''
        if c in s:
            result.append(s[: s.index(c)])
            break
        else:
            result.append(s)
    return ''.join(result)

def img_from_base64(imagestring):
    try:
        jpgbytestring = base64.b64decode(imagestring)
        nparr = np.frombuffer(jpgbytestring, np.uint8)
        r = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return r
    except ValueError:
        return None
    
def downsample_and_save(img, save_path, max_size=512):
    h, w = img.shape[:2]
    
    # Compute scale factor to keep aspect ratio
    scale = min(max_size / h, max_size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize image
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Save
    cv2.imwrite(save_path, resized)

def load_linelist_file(linelist_file):
    if linelist_file is not None:
        line_list = []
        with open(linelist_file, 'r') as fp:
            for i in fp:
                line_list.append(int(i.strip()))
        return line_list

class TSVFile(object):
    def __init__(self, tsv_file, generate_lineidx=False):
        self.tsv_file = tsv_file
        self.lineidx = op.splitext(tsv_file)[0] + '.lineidx'
        self._fp = None
        self._lineidx = None
        # the process always keeps the process which opens the file. 
        # If the pid is not equal to the currrent pid, we will re-open the file.
        self.pid = None
        # generate lineidx if not exist
        if not op.isfile(self.lineidx) and generate_lineidx:
            generate_lineidx(self.tsv_file, self.lineidx)

    def __del__(self):
        if self._fp:
            self._fp.close()

    def __str__(self):
        return "TSVFile(tsv_file='{}')".format(self.tsv_file)

    def __repr__(self):
        return str(self)

    def num_rows(self):
        self._ensure_lineidx_loaded()
        return len(self._lineidx)

    def seek(self, idx):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        try:
            pos = self._lineidx[idx]
        except:
            print('{}-{}'.format(self.tsv_file, idx))
            raise
        self._fp.seek(pos)
        return [s.strip() for s in self._fp.readline().split('\t')]

    def seek_first_column(self, idx):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        pos = self._lineidx[idx]
        self._fp.seek(pos)
        return read_to_character(self._fp, '\t')

    def get_key(self, idx):
        return self.seek_first_column(idx)

    def __getitem__(self, index):
        return self.seek(index)

    def __len__(self):
        return self.num_rows()

    def _ensure_lineidx_loaded(self):
        if self._lineidx is None:
            print('loading lineidx: {}'.format(self.lineidx))
            with open(self.lineidx, 'r') as fp:
                self._lineidx = [int(i.strip()) for i in fp.readlines()]

    def _ensure_tsv_opened(self):
        if self._fp is None:
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()

        if self.pid != os.getpid():
            print('re-open {} because the process id changed'.format(self.tsv_file))
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()

def get_body_parts_from_vertices(vertices_list, threshold=0.1):
    vertices_set = set(vertices_list)
    return [part for part, part_vertices in MERGED_SEGM.items()
            if len(vertices_set.intersection(set(part_vertices))) / len(part_vertices) >= threshold]

def main():
    img_tsv = TSVFile(TSV_IMG_FILE)
    label_tsv = TSVFile(TSV_LABEL_FILE)

    body_model = build_layer(BODY_MODEL_PATH, model_type="smplh", use_pca=False, gender='neutral', num_betas=10)
    body_faces = torch.from_numpy(body_model.faces.astype(np.int32))
    body = body_model(body_pose=VIRTUVIAN_POSE)
    vertices = body.vertices[0].detach()
    vertex_normals = compute_vertex_normals(vertices, body_faces)
    vertex_colors = (vertex_normals + 1) / 2
    mesh = Meshes(
        verts=[vertices.to(device)],
        faces=[body_faces.to(device)],
        textures=TexturesVertex(verts_features=vertex_colors.unsqueeze(0).to(device))
    )

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    mask_dir = os.path.join(OUTPUT_ROOT, 'hcontact_vitruvian')
    img_dir = os.path.join(OUTPUT_ROOT, 'images')
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    body_parts_name = {}
    img_list = []
    contact_vertices_list = {}

    for idx in tqdm(range(len(img_tsv))):
        key = img_tsv.get_key(idx)
        imgname = os.path.basename(key)

        key_split = key.split('/')
        unique_key = f'{key_split[8]}_{key_split[10]}_{key_split[11]}_{imgname}'

        annotations = json.loads(label_tsv[idx][1])[0]
        contact = np.array(annotations['contact']).reshape(-1)
        contact_vertices = np.where(contact == 1.0)[0]
        part_names = get_body_parts_from_vertices(contact_vertices)

        body_parts_name[unique_key] = part_names
        if len(part_names) == 0:
            print(f"[Warning] No contact vertices found for {unique_key}, skipping...")
            continue

        img_row = img_tsv[idx]
        rgb = img_from_base64(img_row[-1])
        if rgb is not None:
            downsample_and_save(rgb, os.path.join(img_dir, unique_key))
        else:
            print(f"[Warning] Failed to decode image at index {idx}")
            continue
            
        for viewname, cam_params in VIEWS.items():
            out_path = os.path.join(mask_dir, f"{unique_key[:-4]}_{viewname}.png")
            if os.path.exists(out_path):
                continue

            mask, _, _ = project_vertices_and_create_mask(
                mesh, cam_params, contact_vertices, image_size=RENDER_IMG_SIZE
            )
            cv2.imwrite(out_path, mask)

        img_list.append(unique_key)
        contact_vertices_list[unique_key] = contact


    jl.dump(body_parts_name, os.path.join(OUTPUT_ROOT, 'body_parts_train.pkl'))
    jl.dump(img_list, os.path.join(OUTPUT_ROOT, 'img_list_train.pkl'))
    jl.dump(contact_vertices_list, os.path.join(OUTPUT_ROOT, 'contact_vertices_train.pkl'))

if __name__ == '__main__':
    main()