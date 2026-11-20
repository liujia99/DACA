import os
import sys
import re
from tqdm import tqdm
import argparse
import torch

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from datasets.scanref import *
from datasets.nr3d import *
from mm_utils.utils import *
from agent.tools.sub_tools import proposal_matching

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--cache_dir', type=str, default='3dagent_workspace/daca')
    parser.add_argument('--result_dir', type=str, default='3dagent_result/daca')

    parser.add_argument('--scannet_video_path', type=str, default='/data/users/jmliu/data/scannet-frames')
    parser.add_argument('--scannet_info_path', type=str, default='/data/users/jmliu/data/scannet-dataset')

    parser.add_argument('--client_id', type=str, default='/data/users/jmliu/model/Qwen3-VL-32B-Instruct')
    parser.add_argument('--sam_path', type=str, default='/data/users/jmliu/model/sam3')

    parser.add_argument('--data_dir', type=str, default='/data/users/jmliu/data')
    parser.add_argument('--dataset_name', type=str, default='SCANREF', choices=['SCANREF', 'NR3D'])

    parser.add_argument('--max_steps', type=int, default=30)
    parser.add_argument('--ste', type=str2bool, default=False, help='Semantic temporal extensions -> ExpandFromSecondaryViewTool')
    parser.add_argument('--mge', type=str2bool, default=False, help='Multi-View geometric extensions -> CentroidCompleteTool')

    parser.add_argument('--mg_ste', type=str2bool, default=True, help='Semantic temporal extensions -> ExpandFromSecondaryViewToolV2')
    parser.add_argument('--ka_mge', type=str2bool, default=True, help='Multi-View geometric extensions -> CentroidCompleteToolV2')


    parser.add_argument('--skill_name', type=str, default='SKILL_DACA.md')
    args = parser.parse_args()
    return args


def getitem(item, anno_path):
    scene_id = item["scene_id"]
    object_id = int(item["object_id"])
    object_name = item["object_name"]
    object_name = object_name.replace('_', ' ')
    description = item["description"].strip()

    ply_path = os.path.join(anno_path+f'/{scene_id}', scene_id+'_vh_clean_2.ply')
    segs_path = os.path.join(anno_path+f'/{scene_id}', scene_id+'_vh_clean_2.0.010000.segs.json')
    agg_path = os.path.join(anno_path+f'/{scene_id}', scene_id+'.aggregation.json')

    # axis alignment matrix
    axisAlignment = read_axis_alignment(os.path.join(anno_path+f'/{scene_id}', scene_id+'.txt'))

    # extract layout
    labels, bboxes = load_bboxes(ply_path, segs_path, agg_path, axisAlignment)
    cnt = labels.count(object_name)
    assert len(labels) == len(bboxes)
    labels = [labels[object_id]]
    bboxes = [bboxes[object_id]]

    # load point cloud
    point_cloud = load_o3d_pcd(ply_path)
    # axis alignment
    points = np.asarray(point_cloud.points)
    points_aligned = align_axis(points, axisAlignment)
    point_cloud.points = o3d.utility.Vector3dVector(points_aligned)
    
    data_dict = {'bbox': []}
    for label, bbox in zip(labels, bboxes):
        x, y, z, sx, sy, sz = bbox
        data_dict['bbox'].append(bbox)
    data_dict['bbox'] = torch.tensor(np.array(data_dict['bbox']))

    # get frames
    data_dict['scene_id'] = scene_id
    data_dict['point_cloud'] = point_cloud
    data_dict['object_name'] = labels
    data_dict['object_id'] = object_id
    data_dict['description'] = description
    data_dict['axisAlignment'] = axisAlignment
    data_dict['type'] = 'multiple' if cnt > 1 else 'unique'
    data_dict['easy_or_hard'] = item['easy_or_hard']
    data_dict['dep_or_indep'] = item['dep_or_indep']
    
    return data_dict


# ====================== 多显卡+单程序并行 ======================
def run_agent(args):
    os.makedirs(args.result_dir, exist_ok=True)

    from agent.loop import AgentLoop

    # ====================== 关键：模型自动使用所有可见GPU ======================
    agent = AgentLoop(
        client_id=args.client_id,
        sam_path=args.sam_path,
        cache_dir=args.cache_dir,
        scannet_video_path=args.scannet_video_path,
        scannet_info_path=args.scannet_info_path,
        max_steps=args.max_steps,
        args=args
    )

    output_file = os.path.join(args.result_dir, f"{args.dataset_name}_results.json")

    if args.dataset_name == 'SCANREF':
        object_lookup_table_path = os.path.join(args.data_dir, 'seeground_object_lookup_table/scanrefer/')
        split_path = os.path.join(args.data_dir, 'scanref/scanrefer_val_250_refined.json')
        dataset = ScanRefDataset(
            split_path=split_path,
            anno_path=args.scannet_info_path,
            video_path=args.scannet_video_path,
            num_frames=300,
        )
    else:
        object_lookup_table_path = os.path.join(args.data_dir, 'seeground_object_lookup_table/nr3d/')
        split_path = os.path.join(args.data_dir, 'referit3d/nr3d_val_250_refined.json')
        dataset = Nr3DDataset(
            split_path=split_path,
            anno_path=args.scannet_info_path,
            video_path=args.scannet_video_path,
            num_frames=300,
        )

    # 初始化结果列表
    results = []
    # 如果文件已存在，加载已有结果（用于去重判断）
    if os.path.exists(output_file):
        results = load_json(output_file)

    # 遍历数据集
    for item in tqdm(dataset):
        item = getitem(item, args.scannet_info_path)
        gt_bbox = item['bbox'][0].cpu().numpy()
        current_scene_id = item['scene_id']
        current_query = item['description']

        # ======================
        # 核心：检查是否已处理过（scene_id + query 唯一匹配）
        # ======================
        skip = False
        for res in results:
            if res["scene_id"] == current_scene_id and res["query"] == current_query:
                # 🔥 关键：如果是 N/A，强制重新执行，不跳过
                if res.get("pred_bbox", "") == "N/A":
                    print(f"\n🔄 发现失败结果 pred_bbox=N/A，重新执行：scene={current_scene_id}, query={current_query}")
                    skip = False  # 不跳过，要重新跑
                else:
                    print(f"\n✅ 已存在有效结果：scene={current_scene_id}, query={current_query}，跳过")
                    skip = True
                break
        if skip:
            continue

        # ======================
        # 未处理过 → 执行流程
        # ======================
        agent._make_cache_dir()

        try:
            task_success = agent.run(f"The scene_id is {current_scene_id}. The query is: {current_query}")
        except Exception as e:
            print(f"❌ Agent运行失败: {e}")
            task_success = False

        # 解析预测框
        if task_success:
            try:
                chat_history = load_json(os.path.join(args.cache_dir, 'chat_history.json'))
                final_observation = chat_history[-2]['content'][0]['text']
                pred_bbox = np.array(re.findall(r"[-+]?\d*\.\d+|\d+", final_observation), dtype=float)
                if pred_bbox.size != 6:
                    task_success = False
            except Exception as e:
                print(f"❌ 解析bbox失败: {e}")
                task_success = False

        # 任务失败
        if not task_success:
            results.append({
                'scene_id': current_scene_id,
                'object_id': item['object_id'],
                'type': item['type'],
                'easy_or_hard': item['easy_or_hard'],
                'dep_or_indep': item['dep_or_indep'],
                'query': current_query,
                'gt_bbox': gt_bbox.tolist(),
                'pred_bbox': 'N/A',
                'iou': 0,
                'matched_bbox': 'N/A',
                'matched_iou': 0,
            })
            save_json(results, output_file)
            continue

        # 计算IOU + 匹配
        iou = calculate_iou_3d(gt_bbox, pred_bbox)
        match_threshold = 0.25
        scene_pcd = item['point_cloud']
        
        matched_bbox = proposal_matching(
            pred_bbox,
            os.path.join(object_lookup_table_path, f"pred/{current_scene_id}.json"),
            scene_pcd,
            match_threshold
        )
        matched_iou = calculate_iou_3d(gt_bbox, matched_bbox)

        # ======================
        # 🔥 保存结果：自动覆盖旧的 N/A 记录
        # ======================
        # 先删除这个场景+query的旧记录（无论是不是N/A）
        results = [
            r for r in results
            if not (
                    r["scene_id"] == current_scene_id
                    and r["query"] == current_query
            )
        ]

        results.append({
            'scene_id': current_scene_id,
            'object_id': item['object_id'],
            'query': current_query,
            'type': item['type'],
            'easy_or_hard': item['easy_or_hard'],
            'dep_or_indep': item['dep_or_indep'],
            'gt_bbox': gt_bbox.tolist(),
            'pred_bbox': pred_bbox.tolist(),
            'iou': iou,
            'matched_bbox': matched_bbox.tolist(),
            'matched_iou': matched_iou,
        })

        # 每完成一个样本立即保存
        save_json(results, output_file)


if __name__ == "__main__":
    args = parse_args()
    print('args=>', args)
    init_seeds(args.seed)

    # 启动单程序 + 多卡模型并行
    run_agent(args)