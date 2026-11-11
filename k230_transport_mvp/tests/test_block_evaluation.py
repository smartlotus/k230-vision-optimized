import unittest
from tools.evaluate_block_labels import evaluate
from tools.sample_block_lab import sample


class EvaluationTests(unittest.TestCase):
    def test_wrong_color_extra_and_missing_are_counted(self):
        report={'samples':[{'frame_id':90,'detections':[
            {'target_id':1,'bbox':[10,10,20,20],'center':[20,20]},
            {'target_id':3,'bbox':[50,10,20,20],'center':[60,20]}]}]}
        labels={'scope':'test','label_source':'synthetic test only','frames':[{'frame_id':90,'objects':[
            {'target_id':1,'bbox':[10,10,20,20]},{'target_id':2,'bbox':[50,10,20,20]}]}]}
        result=evaluate(report,labels)
        self.assertEqual((result['true_positives'],result['false_positives'],result['false_negatives']),(1,1,1))
        self.assertFalse(result['is_generalization_acceptance'])

    def test_missing_frame_does_not_silently_reduce_denominator(self):
        with self.assertRaisesRegex(ValueError,'Missing'):
            evaluate({'samples':[]},{'frames':[{'frame_id':90,'objects':[]}]})

    def test_grid_samples_only_the_selected_native_coordinates(self):
        grid={'x0':2,'y0':2,'stride':4,'lab_grid':[[[10+x+y,-10,-30] for x in range(4)] for y in range(4)]}
        result=sample(grid,{'6':[0,0,16,16]})
        self.assertEqual(result['6']['native_samples'],16)
        self.assertEqual(result['6']['median'][1:],[-10,-30])
        with self.assertRaises(ValueError):sample(grid,{'6':[2,2,1,1]})


if __name__=='__main__':unittest.main()
