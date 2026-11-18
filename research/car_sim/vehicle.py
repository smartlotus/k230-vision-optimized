"""Build the photo-based K230 vehicle in Blender. Geometry helpers use millimetres."""

import json
from math import cos, pi, radians, sin, tan
from pathlib import Path

import bpy
from mathutils import Vector


MM = 0.001


def read_dimensions():
    path = Path(__file__).resolve().with_name("vehicle_dimensions.json")
    return json.loads(path.read_text(encoding="utf-8"))


def material(name, rgb, metal=0.0, rough=0.45):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.diffuse_color = (*rgb, 1.0)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*rgb, 1)
    shader.inputs["Metallic"].default_value = metal
    shader.inputs["Roughness"].default_value = rough
    return mat


class VehicleBuilder:
    def __init__(self):
        self.collection = bpy.data.collections.new("K230 | Chassis and camera rig")
        bpy.context.scene.collection.children.link(self.collection)
        self.carbon = material(
            "Vehicle • graphite laminate", (0.043, 0.052, 0.065), 0.42, 0.43
        )
        self.black = material(
            "Vehicle • black anodised alloy", (0.020, 0.029, 0.040), 0.66, 0.3
        )
        self.darkblue = material(
            "Vehicle • dark-blue camera carrier", (0.012, 0.073, 0.26), 0.27, 0.33
        )
        self.blue = material("Vehicle • blue brackets", (0.036, 0.24, 0.57), 0.18, 0.42)
        self.steel = material(
            "Vehicle • machined alloy", (0.53, 0.61, 0.66), 0.84, 0.26
        )
        self.rubber = material(
            "Vehicle • roller elastomer", (0.022, 0.027, 0.034), 0.05, 0.69
        )
        self.ivory = material(
            "Vehicle • nylon and flat cable", (0.84, 0.83, 0.76), 0.05, 0.55
        )
        self.pcb = material("Vehicle • PCB", (0.027, 0.115, 0.075), 0.12, 0.46)
        self.glass = material("Vehicle • lens glass", (0.018, 0.12, 0.17), 0.54, 0.15)

    def obj(self, name, parent, pos, data=None):
        obj = bpy.data.objects.new(name, data)
        self.collection.objects.link(obj)
        obj.parent = parent
        obj.location = Vector(pos) * MM
        if data is None:
            obj.empty_display_type = "PLAIN_AXES"
            obj.empty_display_size = 0.022
        return obj

    def mesh(self, name, parent, pos, verts, faces, mat, bevel=0.0, smooth=False):
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata([Vector(v) * MM for v in verts], [], faces)
        mesh.update()
        obj = self.obj(name, parent, pos, mesh)
        mesh.materials.append(mat)
        if smooth:
            for polygon in mesh.polygons:
                polygon.use_smooth = True
        if bevel:
            spec = obj.modifiers.new("Machined edges", "BEVEL")
            spec.width = MM * bevel
            spec.segments = 2
            spec.harden_normals = True
            obj.modifiers.new("Face normals", "WEIGHTED_NORMAL")
        return obj

    def box(self, name, parent, pos, dims, mat, bevel=0.45):
        x, y, z = [d / 2 for d in dims]
        loop = [(-x, -y), (x, -y), (x, y), (-x, y)]
        verts = [(a, b, c) for c in (-z, z) for a, b in loop]
        faces = [(3, 2, 1, 0), (4, 5, 6, 7)]
        faces += [(i, (i + 1) % 4, (i + 1) % 4 + 4, i + 4) for i in range(4)]
        return self.mesh(name, parent, pos, verts, faces, mat, bevel)

    def beam(self, name, parent, start, end, width, depth, mat, bevel=0.45):
        """Rectangular strut with local +Z along the specified endpoints."""
        a, b = Vector(start), Vector(end)
        obj = self.box(
            name, parent, (a + b) / 2, (width, depth, (b - a).length), mat, bevel
        )
        obj.rotation_euler = (b - a).to_track_quat("Z", "Y").to_euler()
        return obj

    def profile(self, name, parent, pos, rings, axis, mat, sides=24, smooth=False):
        verts = [
            (radius * cos(2 * pi * k / sides), radius * sin(2 * pi * k / sides), z)
            for z, radius in rings
            for k in range(sides)
        ]
        faces = [tuple(reversed(range(sides)))]
        for row in range(len(rings) - 1):
            lower, upper = row * sides, (row + 1) * sides
            for i in range(sides):
                j = (i + 1) % sides
                faces.append((lower + i, lower + j, upper + j, upper + i))
        faces.append(tuple(range(len(verts) - sides, len(verts))))
        obj = self.mesh(name, parent, pos, verts, faces, mat, smooth=smooth)
        obj.rotation_euler = Vector(axis).to_track_quat("Z", "Y").to_euler()
        return obj

    def disk(self, name, parent, pos, radius, depth, axis, mat, sides=24):
        return self.profile(
            name,
            parent,
            pos,
            [(-depth / 2, radius), (depth / 2, radius)],
            axis,
            mat,
            sides,
        )

    def wheel(self, parent, sx, sy, cfg):
        wheel = self.obj(
            f"Mecanum {'R' if sx>0 else 'L'}{'F' if sy>0 else 'B'}",
            parent,
            (
                sx * cfg["wheel_track_centres"] / 2,
                sy * cfg["wheelbase"] / 2,
                cfg["wheel_radius"],
            ),
        )
        self.disk("Black wheel hub", wheel, (0, 0, 0), 23, 26, (1, 0, 0), self.black)
        self.disk(
            "Outer metal flange", wheel, (sx * 13, 0, 0), 24, 3, (1, 0, 0), self.steel
        )
        self.disk("Axle boss", wheel, (sx * 15.7, 0, 0), 7, 2.5, (1, 0, 0), self.steel)
        rollers = []
        for i in range(8):
            t = i * pi / 4
            y, z = cos(t), sin(t)
            axis = (0.74, -sx * sy * 0.67 * z, sx * sy * 0.67 * y)
            roller = self.profile(
                "Diagonal traction roller",
                wheel,
                (0, 26 * y, 26 * z),
                [
                    (-16, 2.4),
                    (-12, 4.6),
                    (-7, 6.6),
                    (0, 7.8),
                    (7, 6.6),
                    (12, 4.6),
                    (16, 2.4),
                ],
                axis,
                self.rubber,
                12,
                True,
            )
            rollers.append(roller)
            self.beam(
                "Roller axle carrier",
                wheel,
                (sx * 13.3, 20 * y, 20 * z),
                (sx * 13.3, 31 * cos(t + 0.22), 31 * sin(t + 0.22)),
                4,
                7,
                self.steel,
            )
            self.disk(
                "Hub fastener",
                wheel,
                (sx * 15.3, 18 * y, 18 * z),
                1.6,
                1.1,
                (1, 0, 0),
                self.black,
                6,
            )
        # Place the tread tangent to the floor using the actual faceted profile.
        low = min(
            (r.rotation_euler.to_matrix() @ v.co).z + r.location.z
            for r in rollers
            for v in r.data.vertices
        )
        wheel.location.z = -low
        return wheel

    def deck(self, root, d, cfg):
        deck_h = d["chassis_top_height"]
        z = deck_h - 2
        # Slotted open decks and corner X-bracing follow the overhead photograph.
        self.box(
            "Lower carbon tray", root, (0, 0, 8.5), (202, 270, 4), self.carbon, 1.5
        )
        half_l = d["chassis_length"] / 2
        for side in (-1, 1):
            self.box(
                "Deck edge",
                root,
                (side * 100, 0, z),
                (8, 2 * half_l - 18, 4),
                self.carbon,
                1,
            )
            self.box(
                "Crossmember",
                root,
                (0, side * (half_l - 4.5), z),
                (180, 9, 4),
                self.carbon,
                1,
            )
            self.box(
                "Transverse upper girder",
                root,
                (0, side * 57, z),
                (202, 9, 4),
                self.carbon,
            )
            self.box(
                "Central upper girder",
                root,
                (side * 45, 0, z),
                (9, 278, 4),
                self.carbon,
            )
            for s in (-1, 1):
                self.beam(
                    "Clipped deck corner",
                    root,
                    (side * 100, s * (half_l - 10), z),
                    (side * 89, s * (half_l - 4), z),
                    9,
                    4,
                    self.carbon,
                )
                self.beam(
                    "Corner diagonal",
                    root,
                    (side * 46, s * 61, z),
                    (side * 93, s * (half_l - 9), z),
                    7,
                    4,
                    self.carbon,
                )
                self.beam(
                    "Corner cross brace",
                    root,
                    (side * 95, s * 62, z),
                    (side * 46, s * (half_l - 9), z),
                    7,
                    4,
                    self.carbon,
                )
                self.box(
                    "Blue motor housing",
                    root,
                    (side * 75, s * 99, 43),
                    (51, 50, 61),
                    self.blue,
                    3,
                )
                self.box(
                    "Motor housing foot",
                    root,
                    (side * 75, s * 99, 16),
                    (58, 57, 6),
                    self.blue,
                    1.6,
                )
                self.disk(
                    "Motor can",
                    root,
                    (side * 47, s * 99, 42),
                    13,
                    19,
                    (1, 0, 0),
                    self.steel,
                )
                self.disk(
                    "Tray standoff",
                    root,
                    (side * 97, s * 62, 43),
                    2.2,
                    62,
                    (0, 0, 1),
                    self.steel,
                )
                self.disk(
                    "Countersunk deck screw",
                    root,
                    (side * 97, s * 62, deck_h + 0.35),
                    2.0,
                    0.7,
                    (0, 0, 1),
                    self.black,
                    12,
                )
            self.beam(
                "Open deck diagonal",
                root,
                (side * 45, -57, z),
                (-side * 45, 57, z),
                12,
                4,
                self.carbon,
            )
        self.box(
            "Underslung electronics block",
            root,
            (-40, -26, 32),
            (48, 78, 37),
            self.black,
            2,
        )
        self.box("Base controller board", root, (33, -10, 30), (57, 69, 2), self.pcb)
        self.box("Controller package", root, (32, -3, 34), (20, 26, 5), self.black)
        cable = [
            (45, -21, 34),
            (62, -8, 51),
            (63, 10, 69),
            (52, 26, 82),
            (42, 39, 84),
            (35, 53, 74),
        ]
        for a, b in zip(cable, cable[1:]):
            self.beam("Ribbon cable", root, a, b, 11, 0.9, self.ivory, 0.1)
        return [self.wheel(root, sx, sy, cfg) for sx in (-1, 1) for sy in (-1, 1)]

    def camera_rig(self, root, d, cfg, right_mid):
        top = d["chassis_top_height"]
        pedestal = d["pedestal_rise"]
        length = d["overall_height"] - top
        span = d["mast_span"]
        # At deployment the local +Y boom points right. Preserve its 180 mm
        # radius while centering the lens fore/aft between both right hubs.
        assert cfg["deploy_yaw_deg"] == -90
        side = cfg["turret_y"] - right_mid.y / MM
        radius = self.confirmed["pivot_to_lens_horizontal_mm"]
        reach = (radius**2 - side**2)**0.5
        driving_x = d["chassis_width"]/2 + cfg["driving_overhang_mm"]
        turret = self.obj("AZIMUTH | turntable", root, (driving_x-reach, cfg["turret_y"], top))
        self.disk("Base ring", turret, (0, 0, 4), 39, 8, (0, 0, 1), self.black, 48)
        self.disk("Turntable rim", turret, (0, 0, 10), 36, 4, (0, 0, 1), self.steel, 48)
        self.box("Turntable plate", turret, (0, 0, 17), (64, span, 7), self.carbon, 2)
        self.box(
            "Mast pedestal", turret, (0, 0, pedestal - 5), (48, span, 10), self.black, 1
        )
        rail_y = (span - 8) / 2
        for s in (-1, 1):
            self.box(
                "Guide tower rail",
                turret,
                (0, s * rail_y, (pedestal + length) / 2),
                (d["mast_depth"], 8, length - pedestal),
                self.black,
                0.7,
            )
            self.box(
                "Steel guide track",
                turret,
                (6.9, s * rail_y, (pedestal + length - 10) / 2),
                (2.2, 4.2, length - pedestal - 10),
                self.steel,
                0.3,
            )
            self.box(
                "Top bearing block",
                turret,
                (0, s * rail_y, length - 7.5),
                (18, 9, 15),
                self.ivory,
                1,
            )
            self.beam(
                "Mast diagonal support",
                turret,
                (-23, s * rail_y, pedestal),
                (-4, s * rail_y, pedestal + 82),
                6,
                5,
                self.black,
            )
            self.box(
                "Lift belt",
                turret,
                (2, s * 11, (length + pedestal) / 2),
                (6, 2.2, length - pedestal - 18),
                self.rubber,
                0.2,
            )
            for h in (53, 95, 137, 179, 221, 263):
                self.disk(
                    "Rail fastener",
                    turret,
                    (8.5, s * rail_y, h),
                    1.75,
                    1.5,
                    (1, 0, 0),
                    self.black,
                    8,
                )
        self.box(
            "Top bridge",
            turret,
            (0, 0, length - 7),
            (14, span - 10, 14),
            self.carbon,
            1,
        )
        self.disk(
            "Upper belt pulley",
            turret,
            (0, 0, length - 14),
            11,
            16,
            (1, 0, 0),
            self.steel,
            24,
        )
        self.disk("Belt motor", turret, (-12, 0, 43), 11, 22, (1, 0, 0), self.steel, 24)
        lift = self.obj("LIFT | camera carriage", turret, (0, 0, 0))
        # Lift's local Z origin is the optical centre itself.
        arm_start = reach - cfg["visible_boom_mm"]
        holder_height = cfg["upright_mount_height_mm"]
        boom_z = holder_height + 3
        self.box("Carriage endplate",lift,(8,0,boom_z/2),(14,20,boom_z),self.darkblue)
        self.box("Right angle crosshead",lift,((side+8)/2,0,boom_z),
                 (abs(side-8)+12,14,7),self.darkblue)
        self.box("Carrier extension",lift,(side,arm_start/2,boom_z),
                 (14,arm_start+5,7),self.darkblue)
        self.box("Lift overlap proxy",lift,(7,0,-21),(8,13,48),self.black)
        plate_z = cfg["bracket_reference_to_lens_tip"]
        self.box(
            "Sliding bridge", lift, (8, 0, plate_z - 5), (14, span, 19), self.blue, 1.4
        )
        self.box(
            "Carrier front face",
            lift,
            (15, 0, plate_z - 10),
            (4, span - 5, 26),
            self.darkblue,
            0.8,
        )
        self.box(
            "Camera boom",
            lift,
            (side, (reach+arm_start)/2, boom_z),
            (10, reach-arm_start, 3),
            self.blue,
            0.6,
        )
        self.box(
            "K230 board support",
            lift,
            (side, arm_start - 28, boom_z + 4),
            (36, 77, 3),
            self.darkblue,
        )
        self.box(
            "K230 electronics proxy",
            lift,
            (side, arm_start - 28, boom_z + 7),
            (32, 67, 2),
            self.pcb,
        )
        self.box(
            "K230 display proxy",
            lift,
            (side, arm_start - 28, boom_z + 10),
            (35, 70, 3),
            self.black,
            1,
        )
        # The physical end bracket is vertical. The horizontal boom fixes it
        # at the top and the optical housing fixes it at the bottom.
        upright = self.box("Camera holder | vertical",lift,(side,reach,3+holder_height/2),
                           (10,9,holder_height),self.darkblue)
        self.box(
            "Camera housing",
            lift,
            (side, reach, plate_z + 2),
            (21, 22, 10),
            self.darkblue,
            1.5,
        )
        self.disk(
            "Downward lens barrel",
            lift,
            (side, reach, 1.75),
            5.3,
            3,
            (0, 0, 1),
            self.black,
            20,
        )
        self.disk(
            "Lens front glass",
            lift,
            (side, reach, 0.2),
            3.8,
            0.35,
            (0, 0, 1),
            self.glass,
            20,
        )
        data = bpy.data.cameras.new("Road camera optics")
        camera = self.obj("ROAD CAMERA | optical centre", lift, (side, reach, 0), data)
        camera.rotation_euler.x = radians(self.confirmed["camera_tilt_from_down_deg"])
        w, h = cfg["sensor_image_aspect"]
        data.sensor_fit = "HORIZONTAL"
        diag = data.sensor_width * (1 + (h / w) ** 2) ** 0.5
        data.lens = diag / (2 * tan(radians(cfg["camera_diagonal_fov_deg"]) / 2))
        data.clip_start = 0.001
        data.clip_end = 12
        data.display_size = 0.025
        return turret, lift, camera, upright


def build_vehicle():
    """Create one rigid chassis with a two-axis rig. No scene reset or animation."""
    parameters = read_dimensions()
    drawing, assumed = parameters["drawing_mm"], parameters["assumed"]
    builder = VehicleBuilder()
    builder.confirmed = parameters["confirmed"]
    root = builder.obj("K230 | Vehicle ground origin", None, (0, 0, 0))
    root["source_units"] = "mm"
    root["dimension_source"] = parameters["source"]
    wheels = builder.deck(root, drawing, assumed)
    right_wheels = [w for w in wheels if w.location.x > 0]
    assert len(right_wheels)==2
    right_mid = (right_wheels[0].location + right_wheels[1].location)/2
    turret, lift, camera, upright = builder.camera_rig(root, drawing, assumed, right_mid)
    nominal = (
        drawing["chassis_top_height"]
        + drawing["pedestal_rise"]
        + drawing["bracket_above_pedestal"]
        - assumed["bracket_reference_to_lens_tip"]
    )
    rig = dict(
        root=root,
        turret=turret,
        lift=lift,
        camera=camera,
        holder=upright,
        right_wheels=right_wheels,
        wheels=wheels,
        parameters=parameters,
        collection=builder.collection,
        nominal_optical_z_mm=nominal,
    )
    set_vehicle_pose(rig, 0)
    return rig


def set_vehicle_pose(rig, deployed_fraction, optical_height_mm=None):
    """Set yaw and absolute optical height relative to chassis ground, without keys."""
    if not 0 <= deployed_fraction <= 1:
        raise ValueError("deployed_fraction must be within [0, 1]")
    d = rig["parameters"]["drawing_mm"]
    a = rig["parameters"]["assumed"]
    confirmed = rig["parameters"]["confirmed"]
    z = optical_height_mm
    if z is None:
        static = rig["nominal_optical_z_mm"]
        z = static + deployed_fraction * (confirmed["driving_optical_height_mm"] - static)
    if not confirmed["optical_height_min_mm"] <= z <= confirmed["optical_height_max_mm"]:
        raise ValueError(f"Optical height outside confirmed limits: {z}")
    yaw = a["stow_yaw_deg"] + deployed_fraction * (
        a["deploy_yaw_deg"] - a["stow_yaw_deg"]
    )
    rig["turret"].rotation_euler.z = radians(yaw)
    rig["lift"].location.z = (z - d["chassis_top_height"]) * MM
