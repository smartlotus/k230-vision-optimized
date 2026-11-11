"""Summarize recorded native telemetry; keep pool accounting and scene limits explicit."""
import hashlib,json,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'evidence/board/live-20261103'

def summarize():
    raw=(DIRECTORY/'resource-benchmark.json').read_bytes();data=json.loads(raw)
    stages=[]
    for name in ('idle','standalone_menu','standalone_live_TOP','live_threshold_TOP','live_selected_TOP','live_QR_CSI2'):
        rows=[r for r in data['baseline']+data['live'] if r['stage']==name]
        clock='timestamp_ms' if name.startswith('live_') else 'elapsed_ms';first=rows[0][clock]
        rows=[r for r in rows if r[clock]-first>=1000]
        cpu=[r['resources']['cpu_percent'] for r in rows if r['resources']['cpu_percent'] is not None]
        result={'stage':name,'sample_count':len(rows),'warmup_discard_ms':1000,'observed_span_ms':rows[-1][clock]-rows[0][clock],
            'cpu_percent_mean':statistics.mean(cpu) if cpu else None,'cpu_percent_min':min(cpu) if cpu else None,'cpu_percent_max':max(cpu) if cpu else None}
        for group in ('python_heap','system_heap','system_pages','media_mmz'):
            records=[r['resources'][group] for r in rows if r['resources'][group].get('total_bytes')]
            values=[r['used_bytes'] for r in records]
            result[group]={'total_bytes':records[0]['total_bytes'],'used_bytes_mean':statistics.mean(values),'used_bytes_min':min(values),
                'used_bytes_max':max(values),'last_minus_first_bytes':values[-1]-values[0]}
        result['processed_fps']=(rows[-1]['frame_id']-rows[0]['frame_id'])*1000/result['observed_span_ms'] if rows[-1]['frame_id'] else 0
        if name.startswith('live_'):result['processing_ms_mean']=statistics.mean(r['processing_ms'] for r in rows)
        stages.append(result)
    summary={'source':'REAL_K230_NATIVE_RTOS_TELEMETRY','raw_sha256':hashlib.sha256(raw).hexdigest(),'stages':stages,
        'method':'Three 10-second baseline/standalone windows and three 15-second live windows; discard first1s of samples per stage',
        'limits':['os.cpu_usage reports current CanMV RTOS runtime, not entire heterogeneous dual-core SoC.',
                  'Memory denominators are Python heap, system heap, system pages and media MMZ pools. They must not be summed into a physical1GB percentage.',
                  'CSI1 and CSI2 initialized; one selected camera processed per frame. This is not simultaneous five-mode or full mission load.',
                  'No neural model inference is run; KPU utilization has no measured percentage.',
                  'Selected mode filters the nearest candidate bounding region, not independently verified object segmentation.',
                  'Short windows establish observed occupancy and FPS, not a guarantee for all scenes or long-duration memory behavior.'],
        'optimization':'Release native image references and collect every frame. Before-fix MMZ grew up to300MiB; after-fix live windows stayed56.1–56.5MiB.'}
    (DIRECTORY/'resource-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    names={'idle':'空闲脚本','standalone_menu':'脱机菜单','standalone_live_TOP':'脱机实时 TOP','live_threshold_TOP':'实时三视图 TOP','live_selected_TOP':'三视图候选筛选','live_QR_CSI2':'实时 QR / CSI2'}
    lines=['# K230 运行资源实测 · 2026-11-03','','本次测量覆盖当前 CanMV RTOS 系统、双摄初始化后的单模式处理和 IDE 传图，不代表双核整芯片/整车比赛全流程负载。',
      'CPU 使用固件 `os.cpu_usage()`；内存使用 `gc.mem_alloc/free` 与 `gc.sys_heap/sys_page/sys_mmz`。不以 Python 堆代替所有内存，不把各池相加后声称占整块1GB的比例。','',
      '| 状态 | 样本数 | CPU均值/范围 | MMZ均值/池总量 | Python堆峰值/堆总量 | 处理帧率 |',
      '|---|---:|---|---|---|---:|']
    for s in stages:
        m=s['media_mmz'];h=s['python_heap']
        lines.append('| %s | %d | %.1f%% / %d–%d%% | %.1f / %.0f MiB | %.2f / %.2f MiB | %.2f |' %
          (names[s['stage']],s['sample_count'],s['cpu_percent_mean'],s['cpu_percent_min'],s['cpu_percent_max'],m['used_bytes_mean']/1048576,m['total_bytes']/1048576,h['used_bytes_max']/1048576,h['total_bytes']/1048576,s['processed_fps']))
    lines+=['','三视图模式额外做原生二值化、彩色掩膜、LCD显示和1920×360合成JPEG传输，所以比原面板负载高。640×360是相机输出尺寸，三联图横向各一幅同帧图；约4.2–4.5fps是这条调试链路的处理速度，不等于传感器30fps输入或比赛运行模式速度。',
      '', '优化前图像临时副本等待Python GC触发，MMZ短时增长到约300MiB；现在每帧释放引用并回收，相同短窗口观察约56MiB且没有继续增长。原始前后样本都保留。CPU仍约82–84%，不据此声称还有82%的余量。',
      '', '实时模式系统堆约8MiB/64MiB，系统页约32MiB/438.87MiB；完整精确数值及分子分母见 resource-summary.json。KPU没有运行模型，未伪造利用率0%仪表读数。',
      '', '测量时采用既有scene03绿色候选，仅用于工程负载比较；实际场景无独立目标真值，不判定颜色识别准确率、几何标定、运动或整车完成。',
      '', '基线每段10秒、实时每段15秒，各丢弃第一秒样本；有限观测范围内MMZ稳定不替代长期可靠性验收。原生遥测抓取本身也有少量CPU和串口开销。']
    (ROOT/'RESOURCE_USAGE_20261103.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary

if __name__=='__main__':
    print(json.dumps({'stages':len(summarize()['stages']),'report':'RESOURCE_USAGE_20261103.md'}))
