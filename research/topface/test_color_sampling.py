from collections import Counter,defaultdict
import unittest
from scene_config import make_config
from supplement_plan import build_plan,planned_config


class ColorSamplingTests(unittest.TestCase):
    def test_full_generator_colors_are_not_a_function_of_count(self):
        seen=defaultdict(set)
        for i in range(1000):
            c=make_config(20261117,i)
            seen[c['expected_instance_count']].update(o['color_id'] for o in c['objects'])
        for count in range(1,6):self.assertEqual(seen[count],set(range(1,7)))

    def test_counterexamples_cover_all_colors_each_count_each_split(self):
        plan=build_plan();seen=defaultdict(Counter);groups=set();seeds=set();splits=Counter()
        for index in range(240):
            c=planned_config(plan,index);groups.add(c['scene_group_id']);seeds.add(c['random_seed']);splits[c['split']]+=1
            seen[(c['split'],c['expected_instance_count'])][c['objects'][0]['color_id']]+=1
            self.assertEqual(c['base_seed'],20261116)
        self.assertEqual(len(groups),240);self.assertEqual(len(seeds),240)
        self.assertEqual(dict(splits),{'train':120,'val':60,'synthetic_test':60})
        for (split,count),hist in seen.items():
            self.assertEqual(set(hist),set(range(1,7)))
            self.assertEqual(set(hist.values()),{4 if split=='train' else 2})


if __name__=='__main__':unittest.main()
