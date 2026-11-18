# Micro-batch Image Classifier

为视觉部署场景（多台相机）提供的**限时图片分类**服务：FastAPI + CPU 版 PyTorch，
单进程微批（micro-batch）推理。提交图片时给出从受理起算的截止毫秒数；
超时或被取消的帧不会产出结果。

**故障帧定位**：当某帧图片导致 TorchScript 批推理抛错时，worker 会在原计算通道上
对该批每帧做**有界**逐帧重试——能单独推理的帧照常返回分类分数，无法单独推理的帧
标记为 `failed`；即使整个模型都报错，每个订单也会在有限次尝试内终止，不会拖累后续订单。

## 快速开始

```bash
docker compose up --build
# 服务: http://localhost:8000   交互文档: http://localhost:8000/docs
```

异常演示（过曝帧使批推理报错，观察故障帧定位）：

```bash
docker compose --profile anomaly up --build
# 异常模型服务: http://localhost:8001   交互文档: http://localhost:8001/docs
```

本地开发（无需 Docker）：

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/uvicorn app.main:app --port 8000
# 异常模型: MODEL_PATH=app/model_anomaly.pt .venv/bin/uvicorn app.main:app --port 8001
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/predict` | 提交图片（multipart `file`）+ 表单 `deadline_ms`（从受理起算）。返回 `order_id`。满额返回 **429** |
| GET | `/orders/{order_id}` | 按单号查询状态与结果 |
| POST | `/orders/{order_id}/cancel` | 按单号取消；已完成返回 **409** |
| GET | `/stats` | 容量/状态/重试计数（演示与观测用） |
| GET | `/health` | 存活探针 |

示例：

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@samples/sample_gradient.png" -F "deadline_ms=5000"
# {"order_id":"7f6770829f36","status":"queued",...}

curl http://localhost:8000/orders/7f6770829f36
# {"status":"completed","result":{"predicted_class":1,"scores":[...10 个分数...]},
#  "attempts":1,"error":null,...}

curl -X POST http://localhost:8000/orders/7f6770829f36/cancel
```

单号状态机：`queued → processing → completed`；`queued/processing → cancelled|expired`；
`processing → failed`（批推理失败且有界逐帧重试用尽）。
`result.scores` 为 10 类 softmax 分数，与提交单号一一对应；
`attempts` 为该帧参与的模型调用次数（批尝试 + 逐帧重试），`error` 仅在 `failed` 时非空。

## 规则实现（与需求逐条对应）

- **容量 16**：排队中 + 计算中（含等待重试的帧）合计最多 16 张，满额提交返回 429。
- **单批 ≤ 4，等批 ≤ 50ms**：模型空闲时最多等 50ms 凑批，凑满 4 张立即开批。
- **最多一批实际推理**：单个 worker 协程 + 单线程执行器，任何时刻只有一批在做前向计算。
- **排队取消/到期**：从队列移除、释放容量，**不进入模型**（`batch_id` 始终为空）。
- **入批后取消/到期**：立即标记（查询即刻可见）、不供结果；**容量占用到计算通道经过该单**
  （批计算结束或其重试轮次），期间不会因此另开一批（worker 严格串行）。
- **批内其他图片正常完成**，分数对应原单；批推理与逐张推理误差 ≤ 1e-5（实测约 7.5e-9）。
- **迟到结果不恢复取消/过期单**：结果回写只针对仍处于 `processing` 且未过期的单。
- **慢推理不挡 API**：阻塞式前向计算放在独立线程，事件循环始终可用。

### 故障帧定位（批失败后的有界重试）

- **原计算通道重试**：批前向抛错后，worker 在同一执行器上对该批每帧逐张重试，
  每帧最多 `RETRY_ATTEMPTS` 次（默认 1；0 表示关闭重试，整批直接失败）。
- **正常帧按原单返回分数**：逐张重试成功的帧转为 `completed`，分数与单张推理一致（≤1e-5）。
- **无法单独推理的帧标为 `failed`**：带 `error`（最后一次错误摘要），不供结果。
- **整模型报错也有界**：每帧最多 `1 + RETRY_ATTEMPTS` 次模型调用，订单有限次内以
  `failed` 终止，后续订单不会被永久阻塞（见 `/stats` 的 `model_calls` / `retries_started`）。
- **待重试帧继续占容量**；取消/到期的单立即呈终态，且**不参加尚未开始的重试**
  （每次重试前重新检查状态与截止时间）。
- **已开始的重试计算占名额到结束**；迟到结果（含重试结果）不得改写取消/过期等终态。
- **重试期间 API 正常响应**：重试同样在执行器线程上 await，提交/查询/取消不被阻塞；
  其他已受理订单沿用原有容量、截止与凑批规则。

## 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `MAX_CAPACITY` | 16 | 排队+计算中最大图片数 |
| `BATCH_SIZE` | 4 | 单批最大图片数 |
| `BATCH_WAIT_MS` | 50 | 空闲模型凑批等待上限 |
| `RETRY_ATTEMPTS` | 1 | 批失败后每帧的有界逐帧重试次数（0=关闭） |
| `INFERENCE_DELAY_MS` | 0 | **演示用**：在推理线程内注入的人为延迟 |
| `MODEL_PATH` | `app/model.pt` | TorchScript 模型路径（设为 `app/model_anomaly.pt` 切换异常演示模型） |

慢推理演示：`docker compose up` 前把 `docker-compose.yml` 里的 `INFERENCE_DELAY_MS`
改为 `2000`，推理约 2 秒/批，期间提交/查询/取消仍在毫秒级响应（可用 `/stats` 观察）。

## 异常演示模型（可切换）

- `app/model_anomaly.pt`：与正常模型**同架构同权重**（同种子生成，权重指纹一致），
  但当批内存在"过曝"帧（归一化均值 > 0.8）时整批抛 `RuntimeError`，模拟使
  TorchScript 批推理报错的故障相机帧。
- `samples/sample_overexposed.png`：过曝样图。把它与正常样图一起提交（凑进同一批），
  可观察到：整批先失败 → 逐帧重试 → 正常帧 `completed`、过曝帧 `failed`。
- 切换方式：`MODEL_PATH=app/model_anomaly.pt`，或直接使用 compose 的
  `anomaly` profile（端口 8001，见上"快速开始"）。

## 模型与样图

- `app/model.pt`：固定权重 TorchScript 模型（小型 CNN，10 类，输入 64×64 RGB）。
  由 `scripts/make_model.py` 以固定种子生成，权重 SHA-256 指纹见脚本输出，可复现。
- `app/model_anomaly.pt`：异常演示模型（同上，权重相同，另加过曝帧检查）。
- `samples/`：5 张确定性生成的样图（渐变/圆环/条纹/噪声/过曝）。

重新生成：

```bash
python scripts/make_model.py --out app/model.pt
python scripts/make_model.py --variant anomaly --out app/model_anomaly.pt
python scripts/make_samples.py --out samples
```

## 测试

```bash
.venv/bin/pytest tests/ -q
```

覆盖：单张完成、批/单张分数一致性（≤1e-5）、凑批窗口、容量 429、排队取消、
入批取消（立即标记+容量仍占用+不另开批）、完成单取消冲突、排队过期、
计算中过期、迟到结果不复活、慢推理下 API 延迟；
故障帧定位（批失败后正常帧逐帧重试完成、故障帧标 `failed`、分数与单张一致）、
整模型报错的有界终止与 API 响应性、重试前取消不重试、重试中取消持名额且迟到结果不改写、
等待重试期间到期不重试、`RETRY_ATTEMPTS=0` 的整批失败回退行为。

## 已知限制

- 订单记录保存在内存中，重启即丢失；未做持久化与历史订单清理（演示规模）。
- 单进程单 worker；多副本扩展需外加负载均衡与共享状态层，不在本交付范围。
- 取消/到期的单在计算通道经过它之前仍占容量（保守策略：名额只由 worker 在
  计算管线经过该单时释放），该窗口有界（不超过前序帧的重试耗时）。
