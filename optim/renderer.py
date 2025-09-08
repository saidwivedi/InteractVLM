import numpy as np
import torch
from pytorch3d.renderer import (
    BlendParams,
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    SoftSilhouetteShader,
    TexturesVertex,
)
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.transforms import Rotate, axis_angle_to_matrix

from .constants import COLOR_HUMAN_BLUE, COLOR_OBJECT_RED, COLOR_WHITE
import trimesh


class P3DRenderer:
    def __init__(self, img_shape, h_faces, o_faces, camera_params, device="cuda"):
        super().__init__()
        self.img_shape = img_shape
        self.h_faces = h_faces
        self.o_faces = o_faces

        R = torch.tensor(
            [[[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
            device=device,
        )
        T = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
        T = torch.bmm(R, T.unsqueeze(-1)).squeeze(-1)

        self.camera = PerspectiveCameras(
            focal_length=camera_params.focal_length.unsqueeze(0),
            principal_point=camera_params.principal_point.unsqueeze(0),
            image_size=torch.tensor([[img_shape[0], img_shape[1]]], device=device),
            R=R,
            in_ndc=False,
            device=device,
        )
        self.device = device

    def get_human_object_mesh(
        self, h_vertices, o_vertices, color_human=COLOR_OBJECT_RED, color_object=COLOR_HUMAN_BLUE
    ):
        vertex_colors = np.tile(np.array(color_human), (o_vertices.shape[0], 1))
        o_colors = torch.tensor(vertex_colors, dtype=torch.float32)
        o_textures = TexturesVertex(verts_features=o_colors[None].to(self.device))

        vertex_colors = np.tile(np.array([color_object]), (h_vertices.shape[0], 1))
        h_colors = torch.tensor(vertex_colors, dtype=torch.float32)
        h_textures = TexturesVertex(verts_features=h_colors[None].to(self.device))

        o_mesh = Meshes(verts=[o_vertices], faces=[self.o_faces], textures=o_textures)
        h_mesh = Meshes(verts=[h_vertices], faces=[self.h_faces], textures=h_textures)

        return join_meshes_as_scene([o_mesh, h_mesh])


class SSRenderer(P3DRenderer):
    def __init__(self, img_shape, h_faces, o_faces, camera_params):
        super().__init__(img_shape, h_faces, o_faces, camera_params)

        blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=[1.0, 1.0, 1.0])
        raster_settings = RasterizationSettings(
            image_size=(img_shape[0], img_shape[1]),
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
            faces_per_pixel=100,
        )
        self.rasterizer = MeshRasterizer(cameras=self.camera, raster_settings=raster_settings)
        self.renderer = MeshRenderer(
            rasterizer=self.rasterizer, shader=SoftSilhouetteShader(blend_params=blend_params)
        )

    def render(self, vertices: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        # * Initialize each vertex to be white in color.
        if kwargs.get("h_vertices") is not None:
            meshes = self.get_human_object_mesh(
                kwargs["h_vertices"], vertices, color_human=COLOR_WHITE, color_object=COLOR_WHITE
            )
        else:
            verts_rgb = torch.ones_like(vertices)[None]  # (1, V, 3)
            textures = TexturesVertex(verts_features=verts_rgb.to(self.device))
            meshes = Meshes(verts=[vertices], faces=[self.o_faces], textures=textures)

        # * Render depth image
        depth = self.rasterizer(meshes).zbuf[..., [0]]
        valid_pixels = depth != -1
        valid_depth = depth[valid_pixels]
        # * Normalize valid depth values
        normalized_valid_depth = (valid_depth - valid_depth.min()) / (
            valid_depth.max() - valid_depth.min()
        )
        depth[valid_pixels] = normalized_valid_depth

        # * Render silhouette image
        silhouette_image = self.renderer(meshes)

        return silhouette_image, depth


class HPRenderer(P3DRenderer):
    def __init__(
        self,
        img_shape,
        h_faces,
        o_faces,
        camera_params,
    ):
        super().__init__(img_shape, h_faces, o_faces, camera_params)

        lights = PointLights(
            device=self.device,
            ambient_color=[[0.5, 0.5, 0.5]],
            diffuse_color=[[0.3, 0.3, 0.3]],
            specular_color=[[0.2, 0.2, 0.2]],
            location=[[-2.7, -2.7, -2.7]],
        )

        raster_settings = RasterizationSettings(
            image_size=(img_shape[0], img_shape[1]), blur_radius=0.0, faces_per_pixel=1
        )
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.camera, raster_settings=raster_settings),
            shader=HardPhongShader(
                device=self.device,
                cameras=self.camera,
                lights=lights,
            ),
        )

    def render(self, img, h_vertices=None, o_vertices=None):
        # Initialize each vertex to be white in color.

        meshes = self.get_human_object_mesh(h_vertices, o_vertices)

        image = self.renderer(meshes)
        hardP_render = image[..., :3].detach().cpu().numpy()
        alpha = image[..., 3].detach().cpu().numpy()

        # Normalize color values if necessary
        if hardP_render.max() <= 1.0:
            hardP_render = (hardP_render * 255).astype(np.uint8)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)

        # Create a boolean mask from the alpha channel
        valid_mask = (alpha > 0)[..., None]

        # Blend the images
        hardP_render_overlay = np.where(valid_mask, hardP_render, img)

        return hardP_render, hardP_render_overlay

    def save_mesh_as_obj(self, h_vertices, o_vertices, filename, separate=True):
        if separate:
            human_vertices = h_vertices.detach().cpu().numpy()
            object_vertices = o_vertices.detach().cpu().numpy()
            human_faces = self.h_faces.detach().cpu().numpy()
            object_faces = self.o_faces.detach().cpu().numpy()
            trimesh_mesh = trimesh.Trimesh(vertices=human_vertices, faces=human_faces)
            trimesh_mesh.export(str(filename).replace(".obj", "_human.obj"))
            trimesh_mesh = trimesh.Trimesh(vertices=object_vertices, faces=object_faces)
            trimesh_mesh.export(str(filename).replace(".obj", "_object.obj"))

        meshes = self.get_human_object_mesh(h_vertices, o_vertices)

        # Get vertex and face data from the combined mesh
        verts = meshes.verts_packed()
        faces = meshes.faces_packed()

        vertices_np = verts.detach().cpu().numpy()
        faces_np = faces.detach().cpu().numpy()
        trimesh_mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np, process=False)
        trimesh_mesh.export(filename)

    def render_side_view(self, h_vertices=None, o_vertices=None):
        mesh = self.get_human_object_mesh(h_vertices, o_vertices)

        # Define rotation angle (in radians)
        angle = torch.tensor(
            [0, np.pi / 2, 0], device=mesh.device
        )  # Rotate 90 degrees around Y-axis

        # Create a rotation matrix
        R = axis_angle_to_matrix(angle)

        # Create a rotation transform
        rotation = Rotate(R)

        # Apply the rotation to the mesh vertices
        rotated_verts = rotation.transform_points(mesh.verts_padded())

        # Create a new mesh with rotated vertices
        rotated_mesh = mesh.update_padded(new_verts_padded=rotated_verts)

        image = self.renderer(rotated_mesh)
        return image[..., :3].detach().cpu().numpy()
