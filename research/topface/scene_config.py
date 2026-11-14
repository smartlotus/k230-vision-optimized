"""Deterministic, independently seeded V1.1 scene configurations (metres)."""
import hashlib
import numpy as np
from geometry import camera_pose, intrinsics, pixel_to_top, project

COLORS = {1:("red",(.65,.018,.012)),2:("yellow",(.72,.58,.008)),
          3:("blue",(.012,.035,.48)),4:("green",(.012,.32,.025)),
          5:("black",(.008,.008,.009)),6:("light_blue",(.025,.38,.56))}


def camera_partition(height_mm,plane_angle_deg,hfov_deg):
    group=f"H{height_mm:g}-P{plane_angle_deg:g}-F{hfov_deg:g}"
    bucket=int(hashlib.sha256(group.encode()).hexdigest()[:8],16) % 10
    return group,('train' if bucket<8 else 'val' if bucket==8 else 'synthetic_test')


def make_config(base_seed, index, single_qualification=False):
    scene_seed=int(np.random.SeedSequence([base_seed,index]).generate_state(1)[0])
    rng=np.random.default_rng(scene_seed)
    count=1 if single_qualification else index % 6
    height=.25 if single_qualification else [.1,.175,.25,.325,.4][index % 5]
    angle=90. if single_qualification else [80.,85.,90.,95.,100.][(index//5) % 5]
    hfov=65. if single_qualification else [62.,65.,68.][(index//25) % 3]
    pose=camera_pose(height,angle,float(rng.uniform(0,360)))
    K=intrinsics(640,360,hfov)
    center=pixel_to_top([320,180],K,pose)[:2]
    right=pose[:2,0]/np.linalg.norm(pose[:2,0])
    down=np.array([right[1],-right[0]])
    # A 3x2 grid guarantees nonintersecting 50mm bases at the shortest height.
    # Some 100mm/five-object views are intentionally truncated, never dropped.
    max_gap=max(.055,min(.12,height*.3))
    gap_x=float(rng.uniform(.055,max_gap));gap_y=float(rng.uniform(.055,max_gap))
    grid=np.array([[x,y] for y in (-gap_y/2,gap_y/2) for x in (-gap_x,0.,gap_x)])
    rotation=float(rng.uniform(-3,3) if height==.1 else rng.uniform(-25,25))
    theta=np.radians(rotation)
    rotate=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    grid=grid@rotate.T
    rng.shuffle(grid)
    objects=[]
    # Separate RNG stream preserves scene geometry while preventing color from
    # sharing index%6 with object count. Frozen legacy inputs remain immutable.
    color_rng=np.random.default_rng(np.random.SeedSequence([base_seed,index,0xC010]))
    color_order=color_rng.permutation(np.arange(1,7)).tolist()
    for instance in range(count):
        if single_qualification:
            xy=center
        else:
            delta=grid[instance]+rng.uniform(-.0006,.0006,2)
            xy=center+delta[0]*right+delta[1]*down
            if count==1:
                # Small groups also cover image corners and the optical centre.
                uv=([320,180],[35,35],[605,35],[35,325],[605,325])[index % 5]
                xy=pixel_to_top(uv,K,pose)[:2]
        color_id=1 if single_qualification else color_order[instance]
        objects.append({"instance_id":instance+1,"color_id":color_id,
            "color_material":COLORS[color_id][0],"position_m":[float(xy[0]),float(xy[1]),0.],
            "yaw_rad":float(rng.uniform(0,2*np.pi)),
            "roughness":float(rng.uniform(.35,.85)),"brightness":float(rng.uniform(.88,1.08))})
    for i,one in enumerate(objects):
        for two in objects[i+1:]:
            if np.linalg.norm(np.array(one['position_m'])[:2]-np.array(two['position_m'])[:2])<.052:
                raise ValueError("Scene layout intersects original block bases")
    group=f"SG-{base_seed}-{index:06d}"
    camera_group,split=camera_partition(height*1000,angle,hfov)
    return {"schema":"topface-scene-v1.1","scene_group_id":group,"random_seed":scene_seed,
        "base_seed":base_seed,"scene_index":index,"split":split,"camera_group_id":camera_group,
        "color_policy":"independent seeded permutation; separate RNG stream from count/layout",
        "width":640,"height":360,"expected_instance_count":count,"max_instances":5,
        "camera_intrinsics":K.tolist(),"camera_pose_world_from_cv":pose.tolist(),
        "camera_height_above_top_mm":height*1000,"plane_angle_deg":angle,"hfov_deg":hfov,
        "intrinsics_status":"INFERRED","distortion":"none; calibrated coefficients UNKNOWN",
        "object_pose_status":"upright","objects":objects,
        "occluder":bool(count and index % 9==4 and not single_qualification),
        "hard_negative":bool(index % 6==0 or index % 11==3) and not single_qualification,
        "background_rgb":rng.uniform(.08,.6,3).tolist(),
        "light_energy":float(rng.uniform(2,12)),
        "light_position":[float(rng.uniform(-.4,.4)),float(rng.uniform(-.4,.4)),float(rng.uniform(.55,1.))],
        "light_size":float(rng.uniform(.15,.5)),"world_strength":float(rng.uniform(.04,.15)),
        "exposure_stops":float(rng.uniform(-.4,.4)),
        "coverage_exception":"Strong out-of-range camera tilts are excluded from normal scenes by the user's 80-100deg envelope"}
