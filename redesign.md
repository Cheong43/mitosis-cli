基于 Beam Search 的 ReAct Agent 设计

1. 设计目标

将 Beam Search 与 ReAct 模式结合，使智能体在每一步探索多个可能的“思考-行动”路径，并保留评分最高的 k 条路径（beam width）继续推进。这种设计可以克服线性 ReAct 容易陷入局部最优的缺点，通过并行探索多条推理链来提升任务完成质量。

---

2. 核心概念

· Beam：当前正在探索的候选路径集合，每个路径是一个状态序列（包含历史思考、行动、观察）。
· 扩展（Expansion）：对 beam 中的每一条路径，生成多个下一步候选（思考-行动对）。
· 评分（Scoring）：为每个候选路径计算一个分数，反映其当前的成功可能性或任务进度。
· 剪枝（Pruning）：从所有扩展出的候选路径中，选取分数最高的 k 条作为下一轮的 beam。

---

3. 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    BeamSearchReActAgent                  │
├─────────────────────────────────────────────────────────┤
│ - beam: List[Path]            // 当前候选路径            │
│ - beam_width: int             // k                       │
│ - max_depth: int              // 最大探索步数             │
│ - llm: LLMInterface           // 大模型调用接口           │
│ - tools: ToolSet              // 可用工具集               │
│ - scorer: PathScorer          // 路径评分器               │
├─────────────────────────────────────────────────────────┤
│ + run(initial_state): Result  // 主入口                  │
│ + expand(path): List[Path]    // 扩展单个路径             │
│ + score(path): float          // 评分                    │
│ + prune(candidates): List     // 保留 top-k              │
│ + terminate(path): bool       // 终止条件判断             │
└─────────────────────────────────────────────────────────┘
```

---

4. 核心算法流程

```
function run(initial_state):
    // 初始化 beam，包含一条初始路径
    beam = [Path(initial_state)]
    for step in 1..max_depth:
        candidates = []
        for path in beam:
            // 如果该路径已经完成，直接加入候选（不再扩展）
            if terminate(path):
                candidates.append(path)
                continue
            // 扩展当前路径，生成多个下一步候选
            expanded = expand(path)
            candidates.extend(expanded)
        
        // 如果所有路径都已终止，提前结束
        if all(terminate(p) for p in candidates):
            return select_best(candidates)
        
        // 评分并保留 top-k
        beam = prune(candidates)
    
    // 达到最大深度，返回当前 beam 中最好的路径
    return select_best(beam)
```

5. 关键组件设计

5.1 路径表示（Path）

每个路径记录：

· state：当前状态，包含对话历史、工具结果、任务上下文等。
· history：已经执行的步骤列表（思考、行动、观察）。
· score：当前路径的评分（可选，也可以实时计算）。
· metadata：如深度、父路径引用（用于回溯）。

建议使用不可变数据结构（如 Immer 或持久化数据结构），以便在扩展时高效共享未修改的部分。

5.2 扩展（Expand）

扩展一个路径时，需要生成多个“下一步”候选。常见做法：

· 单次 LLM 调用生成多个候选行动：在 prompt 中要求 LLM 输出 n 个可能的下一步（思考 + 行动）。例如：
  ```
  请列出接下来可能的 3 种不同行动，每种包含思考过程和具体动作。
  输出格式：
  1. 思考：... 行动：...
  2. 思考：... 行动：...
  ```
  这种方式只需一次 LLM 调用，但需要模型支持生成多个候选。
· 多次 LLM 调用（带随机性）：对同一状态多次调用 LLM（temperature > 0），每次得到一个候选。适合探索多样性，但成本较高。
· 行动空间枚举：对于某些确定性行动（如 API 参数选择），可以直接枚举可能的行动组合。

扩展实现示例：

```python
def expand(path):
    candidates = []
    state = path.state
    # 调用 LLM 生成多个候选思考-行动对
    thought_actions = llm.generate_multiple(state, n=expansion_factor)
    for ta in thought_actions:
        # 执行行动，获得观察
        observation = execute_tool(ta.action)
        # 更新状态
        new_state = state.append(ta.thought, ta.action, observation)
        # 创建新路径
        new_path = Path(
            state=new_state,
            history=path.history + [(ta.thought, ta.action, observation)],
            depth=path.depth + 1
        )
        # 可选：立即计算评分
        new_path.score = scorer.score(new_path)
        candidates.append(new_path)
    return candidates
```

5.3 评分器（PathScorer）

评分器是 beam search 的核心，决定了哪些路径会被保留。评分可以基于：

· 任务进度：是否获得关键信息、中间结果完成度。
· LLM 自评：让 LLM 对当前路径的成功可能性打分（如 0-10 分），并解释理由。
· 启发式指标：如代码执行通过率、API 返回数据质量、与最终目标的距离。
· 综合指标：加权组合多个信号。

设计要点：

· 评分应快速，避免在评分中再次调用昂贵的 LLM（除非必要）。
· 评分可以增量计算：在扩展时即可计算新路径的分数，避免重复计算历史部分。
· 可以引入置信度：若 LLM 生成候选时自带置信度，可以直接利用。

示例评分器：

```python
class PathScorer:
    def score(self, path):
        # 1. 任务特定信号
        progress = self.task_progress(path.state)
        # 2. LLM 自评（可选，较慢）
        confidence = self.llm_confidence(path.state) if need_llm else 0.5
        # 3. 路径长度惩罚（防止过长）
        length_penalty = -0.01 * path.depth
        return progress * 0.6 + confidence * 0.3 + length_penalty
```

5.4 剪枝（Prune）

剪枝步骤保留分数最高的 k 条路径。需要注意：

· 去重：可能有不同路径到达相同状态，可合并（保留分数最高的）。
· 多样性保持：单纯按分数剪枝可能导致所有路径趋同，可引入多样性奖励（如路径间差异度）。
· 提前终止路径：如果某路径已经完成（例如得到最终答案），应优先保留，即使分数不是最高（任务完成即停止）。

剪枝函数实现：

```python
def prune(candidates):
    # 先按分数排序
    candidates.sort(key=lambda p: p.score, reverse=True)
    # 去重（基于状态哈希）
    unique = {}
    for p in candidates:
        key = hash(p.state)  # 需要实现状态哈希
        if key not in unique or p.score > unique[key].score:
            unique[key] = p
    unique_list = list(unique.values())
    unique_list.sort(key=lambda p: p.score, reverse=True)
    # 保留 top-k，同时确保完成路径一定在 beam 中（如果存在）
    completed = [p for p in unique_list if p.is_complete]
    others = [p for p in unique_list if not p.is_complete]
    beam = completed[:k] + others[:k - len(completed)]
    return beam
```

5.5 终止条件（Terminate）

一个路径满足以下任一条件时终止：

· 已经得到明确答案（如最终输出）。
· 达到最大深度。
· 陷入死循环（可通过状态哈希检测）。
· 评分过低（低于阈值，由剪枝阶段处理）。

---

6. 高级设计考虑

6.1 状态表示与快照

由于 beam search 同时维护多条路径，每条路径的状态可能非常庞大（如长对话历史）。为了节省内存：

· 使用持久化数据结构（如 Immutable.js、Immer 的 produce）实现结构共享。
· 或者，每条路径只保存增量差异，共享公共前缀。但这会增加实现复杂度。

6.2 并行扩展

beam 中的每条路径可以并行扩展，充分利用异步 I/O。可以使用 asyncio.gather 或线程池，但要小心共享资源（如 LLM API 限流）。

6.3 观察缓存

如果多个路径尝试执行相同的工具调用（相同参数），可以缓存观察结果，避免重复调用外部 API。缓存键可以是 (tool_name, parameters)。

6.4 评分延迟与即时反馈

评分可以延迟到扩展之后统一进行，也可以在扩展时立即计算。后者能更早剪枝，但可能引入额外开销。权衡后，建议在扩展时计算简单分数（如基于启发式），在剪枝前再综合 LLM 评分（如果需要）。

6.5 平衡探索与利用

· 探索：在扩展时，可以使用较高的 temperature 生成多样化候选。
· 利用：在剪枝时，依赖评分器引导向高潜力路径集中。

可以通过动态调整 beam width 或 temperature 来平衡：前期宽 beam 探索，后期窄 beam 精炼。

---

7. 与普通 ReAct 的区别

特性 普通 ReAct Beam Search ReAct
路径数量 1 k (beam width)
探索策略 贪婪 保留多个候选
状态管理 单一路径 多条路径，需要隔离
决策方式 每次只选一个行动 同时评估多个候选
适用场景 简单、线性任务 复杂、多路径推理任务

---

8. 示例：代码生成任务

假设任务：编写一个 Python 函数，通过单元测试。

初始状态：问题描述。

Beam width = 3, max_depth = 5。

1. 初始 beam = [空路径]。
2. 扩展：LLM 生成 3 个不同的函数草稿（思考+行动）。
3. 执行每个草稿，运行测试，得到观察（通过/失败）。
4. 评分：测试通过率 + 代码复杂度惩罚。
5. 剪枝：保留 3 个最好的候选。
6. 下一轮：对每个候选，LLM 生成修改方案（修复 bug、优化），继续扩展。
7. 直到某个路径所有测试通过，或达到最大深度。

---

9. 潜在挑战与应对

· LLM 调用成本：beam search 会多次调用 LLM，成本线性增长。可通过控制 beam width、使用缓存、复用评分等降低成本。
· 状态爆炸：如果状态非常大（如长文档），需使用紧凑表示（如摘要、关键信息提取）。
· 评分一致性：不同路径的评分可能由不同 LLM 调用生成，需确保评分标准稳定。可使用固定的评分 prompt 或外部验证器。
· 调试困难：多条路径同时推进，日志需详细记录每条路径的决策和分数，并提供可视化工具（如树形图）辅助调试。

---

10. 总结

基于 Beam Search 的 ReAct Agent 通过维护多条候选推理路径，在每一步保留最有潜力的 k 条路径继续探索，有效克服了单一路径贪婪决策的局限性。设计时需重点关注：

· 状态隔离：确保不同路径互不干扰。
· 评分函数：准确反映路径质量，引导搜索方向。
· 扩展策略：如何高效生成多个候选行动。
· 剪枝规则：保留多样性同时聚焦高潜力路径。

这种设计特别适合需要多步推理、存在多种可能解决方案的复杂任务（如编程、数学推理、多轮交互）。通过合理设置 beam width 和深度，可以在计算成本与任务成功率之间取得良好平衡。