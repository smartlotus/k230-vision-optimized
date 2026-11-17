import unittest
import numpy as np
from filter import TemporalGridFilter
from dataset import schedule, audit, build_sequences
from embedded_filter import SoftGridFilter
from competition_paths import seed_start_belief
from localization_runtime import CameraXYTracker
from board_multiroute import RGBWindow
from board_temporal import BenchDecoder
import copy


class BoardInputTests(unittest.TestCase):
    def test_causal_rgb_history_repeats_first_and_clears_stale_windows(self):
        for length in (1, 4, 8):
            window = RGBWindow(np.zeros((3 * length, 96, 160), dtype=np.uint8), length)
            history = []
            for i in range(10):
                frame = np.empty((3, 96, 160), dtype=np.uint8)
                frame[:] = np.array([i * 3, i * 3 + 1, i * 3 + 2])[:, None, None]
                history.append(frame)
                clip = history[-length:]
                window.push(frame)
                np.testing.assert_array_equal(window.packed, np.concatenate([clip[0]] * (length - len(clip)) + clip))
            window.clear()
            frame.fill(37)
            window.push(frame)
            self.assertTrue(np.all(window.packed == 37))

    def test_bench_decoder_forwards_timeouts_and_recovers_in_global_grid(self):
        times = iter(range(0, 10000000, 25000))
        centers = ((1050, 1050), (1050, -1050), (-850, -550))
        latencies = []
        latency = type('Latency', (), {'add': lambda _, value: latencies.append(value)})()
        decoder = BenchDecoder(centers, {'grid_size': 24}, 'filtered', lambda: next(times), int.__sub__, latency)
        decoder([20., 0., -20.], centers, None, None)
        for _ in range(5):
            output = decoder(None, centers, None, None)
        self.assertEqual(output['state'], 'LOST')
        output = decoder([-20., 0., 20.], centers, None, None)
        self.assertTrue(output['relocalized'])
        self.assertEqual(output['candidate_x_mm'], -850.)
        self.assertIsNone(output['camera_x_mm'])
        self.assertEqual(len(latencies), 7)

class FilterTests(unittest.TestCase):
    def test_unknown_motion_retains_global_support(self):
        f = TemporalGridFilter(np.ones((24,24),bool))
        v = np.full((24,24),.001);v[2,2]=1
        f.update(v,0)
        f.update(v,.1)
        self.assertTrue(np.all(f.belief>0))
        self.assertAlmostEqual(f.belief.sum(),1.)

    def test_lost_then_global_relocalizes(self):
        f=TemporalGridFilter(np.ones((24,24),bool),bad_limit=3)
        v=np.zeros((24,24));v[1,1]=1
        f.update(v,0)
        for i in range(3):
            r=f.update(v,(i+1)*.1,visual_quality=0)
        self.assertEqual(r['state'],'LOST');self.assertFalse(r['relocalized'])
        v[:]=0;v[22,22]=1
        r=f.update(v,.4)
        self.assertTrue(r['relocalized']);self.assertEqual(r['state'],'TRACKING')
        self.assertEqual(r['camera_x_mm'],1050)

    def test_timestamp_and_invalid_visual(self):
        f=TemporalGridFilter(np.ones((8,8),bool))
        f.update(np.ones((8,8)),1)
        for ts in (float('nan'),.9):
            with self.assertRaises(ValueError):f.update(np.ones((8,8)),ts)
        self.assertEqual(f.update(np.full((8,8),float('nan')),2)['state'],'DEGRADED')

    def test_hard_gate_requires_bounds_and_can_recover(self):
        f=TemporalGridFilter(np.ones((24,24),bool),hard_gate_enabled=True,bad_limit=2)
        v=np.zeros((24,24));v[1,1]=1
        f.update(v,0)
        with self.assertRaises(ValueError):f.update(v,.1)
        v[:]=0;v[22,22]=1
        f.update(v,.1,100,0)
        self.assertEqual(f.update(v,.2,100,0)['state'],'LOST')
        self.assertTrue(f.update(v,.3,100,0)['relocalized'])


class DatasetTests(unittest.TestCase):
    def test_split_leakage_rejected(self):
        rows=list(schedule('pilot'))
        audit(rows)
        bad=copy.deepcopy(rows)
        bad[1]['split']='test'
        with self.assertRaisesRegex(ValueError,'Split leakage'):audit(bad)

    def test_bad_target_and_timestamp_rejected(self):
        rows=list(schedule('smoke'))
        rows[0]['camera_xy_target']['camera_yaw_deg']=0
        with self.assertRaises(ValueError):audit(rows)
        rows=list(schedule('smoke'));rows[1]['timestamp_s']=float('nan')
        with self.assertRaises(ValueError):audit(rows)

    def test_window_uses_one_trajectory_and_last_target(self):
        rows=list(schedule('pilot'));windows=build_sequences(rows,8)
        self.assertEqual(len(windows),300)
        self.assertEqual(len({w['sequence_id'] for w in windows}),300)
        self.assertTrue(all(w['target_index']==7 for w in windows))


class EmbeddedParityTests(unittest.TestCase):
    def test_top_k_preserves_probability_and_descending_cell_index(self):
        mask = [[True, False, True, True], [True, True, True, True]]
        likelihood = [4., 99., 3., 0., 4., 1., 2., 3.]
        centers = [(x + .5, y + .5) for y in range(2) for x in range(4)]
        for count in (1, 5, 8):
            filt = SoftGridFilter(mask, cell_mm=1., origin=(0., 0.), top_k=count)
            result = filt.update(likelihood, 0.)
            expected = sorted(
                (i for i, valid in enumerate(filt.valid) if valid),
                key=lambda i: (likelihood[i], i),
                reverse=True,
            )[:count]
            self.assertEqual([(cell['camera_x_mm'], cell['camera_y_mm'])
                              for cell in result['top_k']],
                             [centers[i] for i in expected])

    def test_confidence_masses_include_exact_distance_boundaries(self):
        anchors = [(-25., -25.), (125., -25.), (475., -25.)]
        tracker = CameraXYTracker(anchors, startup='global')
        result = tracker.update([2., 1., 0.], 0.)
        self.assertAlmostEqual(result['confidence_local_mass'],
                               sum(tracker.filter.belief[index] for index in tracker.cells[:2]))
        self.assertEqual(result['confidence_far_mass'], tracker.filter.belief[tracker.cells[2]])
        self.assertFalse(result['position_valid'])

    def test_time_scaled_soft_support_matches_masked_scatter(self):
        mask=np.ones((5,8),bool);mask[1:3,2:6]=False
        f=SoftGridFilter(mask.tolist(),process_scale_mm_s=800)
        rng=np.random.default_rng(117)
        visual=np.where(mask,rng.random(mask.shape),0.)
        for dt in (0.,.12,.37,4.0,.37):
            with self.subTest(dt=dt):
                f.initialize(visual.ravel().tolist())
                previous=np.asarray(f.belief).reshape(mask.shape)
                spread=np.zeros_like(previous)
                reach=max(1,min(7,int(np.ceil(800*dt/100))))
                for y,x in zip(*np.nonzero(mask)):
                    window=np.s_[max(0,y-reach):y+reach+1,max(0,x-reach):x+reach+1]
                    allowed=mask[window]
                    spread[window]+=previous[y,x]*allowed/allowed.sum()
                f._predict(dt)
                retain=.95*np.exp(-dt/.5)
                expected=retain*spread+(1-retain)*mask/mask.sum()
                np.testing.assert_allclose(f.prior,expected.ravel(),atol=1e-14)
                self.assertAlmostEqual(sum(f.prior),1.)
                self.assertTrue(np.all(np.asarray(f.prior)[mask.ravel()]>0))

    def test_wider_soft_support_relocalizes_after_invalid_frames(self):
        f=SoftGridFilter([[True]*24 for _ in range(24)],bad_limit=2,process_scale_mm_s=800)
        seed_start_belief(f)
        f.update([1.]*576,0.)
        for timestamp in (.2,1.3):
            result=f.update([0.]*576,timestamp)
        self.assertEqual(result['state'],'LOST')
        visual=[0.]*576
        visual[6*24+3]=1.
        result=f.update(visual,2.)
        self.assertTrue(result['relocalized'])
        self.assertEqual(result['camera_x_mm'],-850.)

    def test_uncalibrated_tracker_labels_output_as_candidate(self):
        tracker=CameraXYTracker([(1050,1050),(1050,-1050)],grid_size=24)
        result=tracker.update([9.,-9.],0.)
        self.assertFalse(result['position_valid'])
        self.assertIsNone(result['camera_x_mm'])
        self.assertEqual(result['candidate_x_mm'],1050)
        self.assertEqual(result['candidate_y_mm'],1050)
        self.assertEqual(result['state'],'DEGRADED')

    def test_blue_prior_starts_bimodal_and_lost_recovery_is_global(self):
        grid=24
        f=SoftGridFilter([[True]*grid for _ in range(grid)],bad_limit=2)
        seed_start_belief(f)
        self.assertAlmostEqual(sum(f.belief[:grid*12]),.5)
        self.assertAlmostEqual(sum(f.belief[grid*12:]),.5)
        first=f.update([1.]*grid**2,0)
        self.assertGreater(first['camera_x_mm'],700)
        self.assertGreater(abs(first['camera_y_mm']),700)
        self.assertTrue(all(value>0 for value in f.belief))
        f.update([0.]*grid**2,.1,visual_quality=0)
        self.assertEqual(f.update([0.]*grid**2,.2,visual_quality=0)['state'],'LOST')
        visual=[0.]*grid**2
        visual[6*grid+3]=1.
        result=f.update(visual,.3)
        self.assertEqual(result['camera_x_mm'],-850)
        self.assertTrue(result['relocalized'])

    def test_topk_never_contains_invalid_zero_mass_cells(self):
        mask=np.zeros((3,3),bool);mask[:2,:2]=True
        visual=np.zeros((3,3));visual[0,0]=1
        for f in (TemporalGridFilter(mask,top_k=4),SoftGridFilter(mask.tolist(),top_k=4)):
            value=visual if isinstance(f,TemporalGridFilter) else visual.tolist()
            result=f.update(value,0.)
            self.assertEqual(len(result['top_k']),4)
            self.assertTrue(all(r['camera_x_mm']<-1000 and r['camera_y_mm']<-1000 for r in result['top_k']))

    def test_cpu_port_matches_reference_with_mask_drop_and_recovery(self):
        mask=np.ones((24,24),bool);mask[:2,:3]=False
        host=TemporalGridFilter(mask);embedded=SoftGridFilter(mask.tolist())
        rng=np.random.default_rng(117);ts=0.
        for i in range(80):
            ts+=.1 if i%7 else .35
            visual=rng.random((24,24));quality=0. if 20<=i<=25 else 1.
            a=host.update(visual,ts,visual_quality=quality)
            b=embedded.update(visual.ravel().tolist(),ts,visual_quality=quality)
            np.testing.assert_allclose(host.belief.ravel(),embedded.belief,rtol=1e-12,atol=1e-15)
            self.assertEqual(a['state'],b['state']);self.assertEqual(a['relocalized'],b['relocalized'])


if __name__=='__main__':unittest.main()
