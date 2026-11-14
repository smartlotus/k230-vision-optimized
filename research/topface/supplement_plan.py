"""Finite240 new count/color counterexamples without replaying old1000."""
from collections import Counter,defaultdict
import numpy as np
from scene_config import make_config,COLORS


def build_plan(base_seed=20261116):
    quotas={'train':24,'val':12,'synthetic_test':12}
    groups=defaultdict(int);rows=[]
    for source_index in range(10000):
        config=make_config(base_seed,source_index);count=config['expected_instance_count'];split=config['split']
        if count==0 or groups[(split,count)]>=quotas[split]:continue
        ordinal=groups[(split,count)];groups[(split,count)]+=1
        first=ordinal%6+1
        rng=np.random.default_rng(np.random.SeedSequence([base_seed,source_index,0xC011]))
        others=rng.permutation([c for c in range(1,7) if c!=first]).tolist()
        rows.append({'source_scene_index':source_index,'expected_instance_count':count,
            'split':split,'camera_group_id':config['camera_group_id'],
            'color_ids':[first]+others[:count-1]})
        if len(rows)==240:break
    if len(rows)!=240:raise ValueError('Could not fill independent split/count quotas')
    # Shuffle output order, retaining source-index identities and camera split.
    np.random.default_rng(base_seed).shuffle(rows)
    return {'schema':'topface-counterexample-scene-plan-v1.1','base_seed':base_seed,'count':240,
        'parent_pilot_run':'k230-topface-v11-pilot-1000-a02','parent_pilot_replayed':False,
        'purpose':'break count/color confounding; every1..5 count includes every first-target color within each split',
        'split_scene_quotas':{split:count*5 for split,count in quotas.items()},'scenes':rows}


def planned_config(plan,output_index):
    row=plan['scenes'][output_index]
    config=make_config(plan['base_seed'],row['source_scene_index'])
    if (config['expected_instance_count'],config['split'],config['camera_group_id'])!=(
            row['expected_instance_count'],row['split'],row['camera_group_id']):
        raise ValueError('Counterexample scene plan no longer matches camera/count geometry')
    if len(row['color_ids'])!=len(config['objects']) or len(set(row['color_ids']))!=len(row['color_ids']):
        raise ValueError('Counterexample per-instance palette is invalid')
    for item,color in zip(config['objects'],row['color_ids']):
        if color not in COLORS:raise ValueError('Unapproved color')
        item.update(color_id=color,color_material=COLORS[color][0])
    config.update(color_policy='split/count-stratified seeded permutation counterexample; geometry unchanged',
                  scene_plan_output_index=output_index,parent_pilot_run=plan['parent_pilot_run'])
    return config
