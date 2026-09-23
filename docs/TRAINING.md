# 数据契约与实验 004 复现

以下命令在项目根目录、已激活实验环境运行。扩展标注已在 datasets/gui-verifier-v1 公开，原图需从上游取得并按数据集README重建；训练适配器尚未公开。自备数据须符合契约。所有输出目录必须是新目录；训练器拒绝覆盖中断的运行，不能把重跑当作无缝恢复。

## 一行数据

```json
{"id":"screen-001","group_id":"session-001","image":"images/001.png","source":"self-owned","request":{"goal":"Verify GUI state","rules":["Use only visible evidence."],"questions":[{"id":"q01","text":"Is a success message visible?","options":[{"id":"yes","text":"Yes"},{"id":"no","text":"No"},{"id":"unknown","text":"Cannot determine from the screenshot"}],"unknown_option_id":"unknown"}]},"labels":{"q01":"yes"},"label_quality":"human","human_reviewed":true}
```

路径相对 JSONL 文件，labels 不进入提示词。相关会话、截图变体、对照对共享 group_id，先分组再切分；校准集独立于训练和选型。实验 004 loader 将题目换成中性编号，去除同图重复题，打乱题序且对齐标签，防止 positive/negative 题号泄漏。

## 本次数据构造

OS-Atlas 桌面/网页与 ScreenQA 移动截图，先冻结来源，过滤已发现的 ScreenSpot 重叠。教师从可见内容生成正负问题对和证据不足问题，再用同一模型盲审；拒绝/异常结果隔离，不强行补标签。最终 5270 对正负问题 + 2326 个 unknown = 12866 个判断。

这属于同模型审核 silver 标签，不是独立人工标注。图文理解教师为 Gemini 3.8 Flash；供应商差异保留在内部审计元数据中，公开源码不依赖原内部平台。当前公开版本仅提供最终契约与训练/评估代码，不包含可直接复跑的教师 API 管线。

训练 yes/no/unknown 为 4205/4205/1856；验证 540/540/234；校准 525/525/236。图片内容 hash 与 group 跨集检查已执行，但缺少独立人工质量审计。

原内部实验文件 SHA-256（公开副本移除了平台任务/供应商字段，内容hash随之变化；请求与标签未变，公开hash见数据集summary.json）：

- train: `e4d244b59aa8e3ed3134629db2cb539fd3ef6b7d08f9f941f85fbb11745ea54b`
- validation: `ed49bf48db659a3882d69e3f723437893deb0b38525dde96d0ff3b83cf7bb818`
- calibration: `efa52d2eb7dde5edff47d7990941b57822ef695584db22391c1116924c89dd64`

## 训练

```bash
python tools/train004.py --model models/Qwen3.5-0.8B \
  --train datasets/gui-verifier-v1/final/train.jsonl --validation datasets/gui-verifier-v1/final/validation.jsonl \
  --output outputs/decision-lr5e5-seed404 --mode decision --lr 5e-5 --seed 404 --epochs 3

python tools/train004.py --model models/Qwen3.5-0.8B \
  --train datasets/gui-verifier-v1/final/train.jsonl --validation datasets/gui-verifier-v1/final/validation.jsonl \
  --output outputs/json-lr5e5-seed404 --mode json --lr 5e-5 --seed 404 --epochs 3
```

另用 `--lr 2e-5` 和不同输出目录重复两种模式。当前不执行 seed405/406。

候选训练 micro-batch=4，有效 batch=8，1284 updates/epoch、共3852步。JSON 每组最多四题，有效 batch=8个序列，485 updates/epoch、共1455步，实际逐序列累积。每约半个 epoch 验证；macro-F1 差距在0.002内优先更低误报成功率，否则更高F1。选定决策 5e-5 step1926；选定 JSON 5e-5 step1455。

输出 `best-adapter/`、`latest-adapter/`、训练配置、逐步日志、验证预测及完成标记。不能拿训练时未合并的分数与最终合并后结果混报。

## 统一评估

分别运行三次，下例省略 adapter 即原始权重：

```bash
python tools/evaluate004.py --model models/Qwen3.5-0.8B \
  --data datasets/gui-verifier-v1/final/validation.jsonl --output outputs/experiment-004/base-full \
  --speed-screens 100 --repeats 5
python tools/evaluate004.py --model models/Qwen3.5-0.8B \
  --adapter outputs/decision-lr5e5-seed404/best-adapter \
  --data datasets/gui-verifier-v1/final/validation.jsonl --output outputs/experiment-004/decision-selected-full \
  --speed-screens 100 --repeats 5
python tools/evaluate004.py --model models/Qwen3.5-0.8B \
  --adapter outputs/json-lr5e5-seed404/best-adapter \
  --data datasets/gui-verifier-v1/final/validation.jsonl --output outputs/experiment-004/json-selected-full \
  --speed-screens 100 --repeats 5
python tools/evaluate004.py --model models/Qwen3.5-0.8B \
  --adapter outputs/decision-lr5e5-seed404/best-adapter \
  --data datasets/gui-verifier-v1/final/calibration.jsonl --output outputs/experiment-004/calibration-eval \
  --methods fast --speed-screens 0
python tools/analyze004.py --root outputs/experiment-004
```

默认方法 `json,json_constrained,logits,fast`。speed-screens=0仅效果评估。K=1/2/4/8 必须存在足够多不同问题，不能复制问题凑数。完整效果评估使用每张图的全部自然问题，速度则使用 K 个问题子集，因此两种报告的格式有效率可能不同。

模型普通和约束生成使用完全相同的 system/user 提示词，见 `src/gui_verifier/experiment004.py`。只有约束生成添加合法数组前缀树。贪心解码、关闭思考、显式 EOS；达到长度上限、非数组、漏答或非法标签均按失败记录。

解析出的完整数组才算有效，不通过解释文本猜答案；格式失败计入严格准确率。另报可解析部分准确率但不作为主指标。所有时间记录保留逐样本结果；各模型分时运行，未做跨模型交错负载控制，因此硬件状态波动仍可能影响小幅差异。

## 复现边界

汇总报告不是原始可审核样本的替代品。目前完整扩展标签及固定划分已公开，截图从上游取得并按manifest重建，适配器未公开。标注请求、标签与划分保持原样，仅移除了内部平台元数据。还未测：独立人工金标、并发4/8、视觉事实翻转对、全部分辨率消融、完整多种子结果。
