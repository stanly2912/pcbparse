# PCB 原理图自动解析

面向赛题六的 PCB 原理图图像解析系统。以 Circuit-IR 为内部表示，经过模型感知、OCR/引脚绑定、物理图构建、电气网络归并、严格校验后，输出官方 JSON。

生产推理只读取图像、模型、配置和静态器件库。输出顶层为 `components`、`pins`、`nets`；缺少器件标识或引脚编号时，不自动编造字段。

## 仓库范围

Git 仅保存主代码及运行、构建所需的配置：

| 路径 | 内容 |
| --- | --- |
| `src/pcbparse/` | 推理、训练、Circuit-IR、校验、序列化、CLI 和服务 |
| `src/data_tools/`、`data_tools/` | 数据准备、训练资产检查和模型晋级工具 |
| `configs/` | 生产流水线、运行 profile 和训练配置 |
| `models/models.lock.json` | 唯一生产模型权威 |
| `models/registry.yaml` | 描述性模型信息 |
| `docker/` | 容器构建和提交入口 |

数据集、权重、完整模型 bundle、测试、评测模块、实验脚本、历史文档及运行结果留在工作区或 llm 服务器，不随 Git 分支分发。评测模块同时排除在安装包之外。当前分支的清理不改写历史提交。

服务器完整项目位于 `/home/data/yinjx/pcbparse-competition/project`，包含合并后的数据与模型资产。新克隆的代码仓库需要另外准备这些资产。

## 安装与检查

运行环境为 Python 3.11 或 3.12。PowerShell 示例：

```powershell
git clone https://github.com/yueryu54111-commits/pcbparse.git
cd pcbparse
python -m pip install -e .
$env:PYTHONPATH = (Resolve-Path './src').Path
python -m pcbparse.cli healthcheck
```

普通健康检查验证进程、导入及序列化。准备锁中指定的完整模型 bundle 后，用实际竞赛配置执行生产就绪检查：

```powershell
python -m pcbparse.cli healthcheck `
  --require-ready `
  --config configs/pipeline.edarace.yaml `
  --models-lock models/models.lock.json
```

模型锁包含 `component`、`ocr`、`pin`、`wire` 四个任务，junction 默认使用确定性 vectorizer。哈希或 lineage 缺失时保持 unready。不得手工修改锁中的 `blessed` 或 `status`。

## 推理与提交

```powershell
python -m pcbparse.cli infer `
  --input ./input `
  --output ./output `
  --profile competition `
  --config configs/pipeline.edarace.yaml `
  --models-lock models/models.lock.json

python -m pcbparse.submission_agent -input ./input -output ./output
```

竞赛入口写出 `output/final_results/<case_id>_Pred.json` 和 `output/decision_log.json`。Linux 容器入口为 `/submission/run.sh`，同时保留兼容入口 `/submission/agent_system`。构建容器前需备齐模型锁指定的 `models/releases/v8-portable/`；官方镜像提供 CUDA 匹配的 PyTorch，`requirements.edarace.txt` 不替换其 torch。

提交镜像从包内安装固定版本的 Linux wheels，并校验其哈希，构建时禁止访问包索引。先在联网准备环境中下载依赖：

```powershell
python -m pip download --index-url https://pypi.org/simple `
  --dest ./work/edarace-wheels --requirement requirements.edarace.txt `
  --only-binary=:all: --no-deps --python-version 3.12 --implementation cp `
  --abi cp312 --abi abi3 --abi none `
  --platform manylinux_2_28_x86_64 --platform manylinux_2_27_x86_64 `
  --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64

python -m data_tools.edarace_bundle --wheelhouse ./work/edarace-wheels `
  --output ./outputs/submissions/new-offline-bundle
```

打包工具核对 PyPI 官方文件哈希、生产模型哈希，使用当前源码预构建项目 wheel。准备环境需已安装 `setuptools>=69` 和 `wheel`；打包过程会访问 PyPI 发布信息。ZIP 中不包含数据集和离线评测代码。使用生成目录中的 `package` 作为 Docker 构建上下文，并在禁止联网时验证镜像：

```powershell
docker build --network=none --pull=false --progress=plain `
  -t pcbparse:edarace-offline ./outputs/submissions/new-offline-bundle/package
```

镜像构建通过后，按官方入口挂载真实 PNG 输入和输出目录进行推理验证。打包成功不会自动标记完整 Docker 构建或推理已通过。

## 训练与工作区验证

训练只使用 `dataset300-v2`，其他派生版本保留审计用途。单 GPU 训练串行执行，先检查初始化资产和配置，再运行 smoke。模型晋级需要完整 bundle 和证据报告，经晋级工具生成新锁。

服务器完整工作区保留 unittest 与离线评测工具。按 unit、integration 顺序执行工作区测试；GitHub CI 只验证代码编译、健康检查和主要入口，不把缺少评测代码的克隆视为完整评测环境。模型加载通过不代表获得官方竞赛成绩。
