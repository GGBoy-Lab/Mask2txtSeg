# -*- coding: utf-8 -*-
"""样本构建流水线（YAML 配置驱动）

流水线共四步，每一步的产物目录如下：

    1. slice    滑窗裁剪影像           -> images/  labels/  meta/  empty/
    2. filter   按背景占比清洗样本      -> images1/ labels1/
    3. to_txt   掩膜 PNG 转 YOLO txt   -> txt/
    4. dataset  划分训练/验证集         -> SINGLECLASSDataset/{images,labels}/{train,val}

所有可变参数（路径、裁剪尺寸、重叠率、后缀、类别与颜色、过滤阈值、验证集比例）
全部写在 YAML 配置文件里，见 configs/ 目录；本脚本不再硬编码任何业务参数。

用法::

    python buid_mask.py                                    # 自动使用 configs/ 下唯一的配置
    python buid_mask.py -c configs/l1_1024.yaml      # 指定配置文件
    python buid_mask.py --list                             # 列出全部可用配置
    python buid_mask.py -c configs/x.yaml --steps slice,to_txt
    python buid_mask.py -c configs/x.yaml --set slice.crop_size=2048
    python buid_mask.py -c configs/x.yaml --set input.out_dir='E:\\out' --dry-run
    python buid_mask.py -c configs/x.yaml --no-progress    # 输出重定向到日志文件时关闭进度条
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = ROOT_DIR / "configs"

# 流水线步骤顺序（不可调整，后一步依赖前一步的产物）
STEP_ORDER = ("slice", "filter", "to_txt", "dataset")
STEP_LABELS = {
    "slice": "影像滑窗裁剪",
    "filter": "样本清洗（背景占比过滤）",
    "to_txt": "掩膜转 YOLO 标签",
    "dataset": "划分训练/验证集",
}

# 默认配置：YAML 中未出现的键将回落到这里的值
DEFAULT_CONFIG = {
    "input": {
        "tif_dir": "",    # 待裁剪影像目录，递归查找 *.tif
        "shp_path": "",   # 标注矢量路径（.shp / .gdb 图层）
        "out_dir": "",    # 样本输出根目录
    },
    "slice": {
        "crop_size": 1024,   # 裁剪窗口边长（像素）
        "overlap": 0.1,      # 滑窗重叠率 [0, 1)
        "suffix": "n",       # 裁剪产物文件名后缀
        "palette": {
            "field": "type",  # shp 中存放类别的字段名
            "classes": [],    # 类别名列表
            "colors": [],     # 与 classes 一一对应的 RGB 颜色
        },
    },
    "filter": {
        "background_threshold": 0.95,  # 背景占比超过该值的样本被丢弃
    },
    "dataset": {
        "val_ratio": 0.1,  # 验证集占比
    },
    "steps": {name: True for name in STEP_ORDER},
}


class ConfigError(Exception):
    """配置文件校验失败。"""


# --------------------------------------------------------------------------- #
# 配置读取与校验
# --------------------------------------------------------------------------- #
def _import_yaml():
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - 仅在缺依赖时触发
        raise SystemExit("缺少 PyYAML 依赖，请先执行：pip install pyyaml") from exc
    return yaml


def deep_merge(base: dict, override: dict) -> dict:
    """递归合并配置，override 中的同名键覆盖 base。"""
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def deep_set(cfg: dict, dotted_key: str, value) -> None:
    """按 "a.b.c" 形式写入嵌套键，缺失的中间层级自动创建。"""
    keys = dotted_key.split(".")
    node = cfg
    for key in keys[:-1]:
        if not isinstance(node.get(key), dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def resolve_config_path(cli_path: str | None) -> Path:
    """定位配置文件：显式指定 > configs/ 下唯一一份配置。"""
    if cli_path:
        candidate = Path(cli_path)
        if not candidate.is_absolute():
            for base in (Path.cwd(), ROOT_DIR):
                if (base / candidate).exists():
                    return (base / candidate).resolve()
        if not candidate.exists():
            raise ConfigError(f"配置文件不存在：{cli_path}")
        return candidate.resolve()

    # 自动发现：忽略以 _ 开头的模板文件
    found = []
    if CONFIG_DIR.is_dir():
        found = sorted(p for p in CONFIG_DIR.iterdir()
                       if p.suffix.lower() in (".yaml", ".yml") and not p.name.startswith("_"))
    if len(found) == 1:
        return found[0]
    if not found:
        raise ConfigError(
            f"未找到配置文件：{CONFIG_DIR} 下没有可用的 *.yaml。\n"
            f"可复制 {CONFIG_DIR / '_template.yaml'} 为一份新配置后再运行。"
        )
    listing = "\n".join(f"  - {p.relative_to(ROOT_DIR)}" for p in found)
    raise ConfigError(f"configs/ 下存在多份配置，请用 -c 指定其中一份：\n{listing}")


def load_config(path: Path) -> dict:
    """读取 YAML 并与默认配置合并。"""
    yaml = _import_yaml()
    with path.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"配置文件根节点必须是映射（字典），当前为 {type(raw).__name__}")
    return deep_merge(DEFAULT_CONFIG, raw)


def _parse_override_value(text: str):
    """把命令行 --set 的值还原成合适的 Python 类型。"""
    yaml = _import_yaml()
    try:
        return yaml.safe_load(text)
    except Exception:
        return text


def validate_config(cfg: dict) -> list[str]:
    """校验配置，返回全部错误信息（一次性报告，避免反复试错）。"""
    errors: list[str] = []
    inp = cfg["input"]
    sl = cfg["slice"]
    palette = sl["palette"]
    enabled = [name for name in STEP_ORDER if cfg["steps"].get(name)]

    if not enabled:
        errors.append("steps 中至少需要开启一个步骤")

    # ---- 路径（仅校验已启用的步骤真正会用到的输入）----
    for key, label in (("tif_dir", "输入影像目录"),
                       ("shp_path", "标注矢量"),
                       ("out_dir", "样本输出目录")):
        if not str(inp.get(key) or "").strip():
            errors.append(f"input.{key}（{label}）不能为空")

    if "slice" in enabled:
        tif_dir = str(inp.get("tif_dir") or "").strip()
        if tif_dir and not Path(tif_dir).is_dir():
            errors.append(f"input.tif_dir 目录不存在：{tif_dir}")
        shp_path = str(inp.get("shp_path") or "").strip()
        if shp_path and not Path(shp_path).exists():
            errors.append(f"input.shp_path 不存在：{shp_path}")

    # ---- 裁剪参数 ----
    crop_size = sl.get("crop_size")
    try:
        if int(crop_size) <= 0:
            errors.append(f"slice.crop_size 必须为正整数，当前为 {crop_size!r}")
    except (TypeError, ValueError):
        errors.append(f"slice.crop_size 必须为正整数，当前为 {crop_size!r}")

    overlap = sl.get("overlap")
    try:
        overlap = float(overlap)
        if not 0.0 <= overlap < 1.0:
            errors.append(f"slice.overlap 需落在 [0, 1) 区间，当前为 {overlap!r}")
    except (TypeError, ValueError):
        errors.append(f"slice.overlap 必须是数字，当前为 {sl.get('overlap')!r}")

    # ---- 类别与颜色 ----
    classes = palette.get("classes")
    colors = palette.get("colors")
    if not isinstance(classes, list) or not classes:
        errors.append("slice.palette.classes 不能为空列表")
    if not isinstance(colors, list) or not colors:
        errors.append("slice.palette.colors 不能为空列表")
    if isinstance(classes, list) and isinstance(colors, list):
        if len(classes) != len(colors):
            errors.append(
                f"slice.palette 类别数与颜色数不一致：classes={len(classes)}，colors={len(colors)}"
            )
        for idx, color in enumerate(colors):
            if not isinstance(color, list) or len(color) != 3:
                errors.append(f"slice.palette.colors[{idx}] 必须是长度为 3 的 RGB 列表，当前为 {color!r}")
                continue
            if not all(isinstance(c, int) and 0 <= c <= 255 for c in color):
                errors.append(f"slice.palette.colors[{idx}] 的取值必须是 0~255 的整数，当前为 {color!r}")
    if not str(palette.get("field") or "").strip():
        errors.append("slice.palette.field（shp 中的类别字段名）不能为空")

    # ---- 阈值类参数 ----
    threshold = cfg["filter"].get("background_threshold")
    try:
        if not 0.0 < float(threshold) <= 1.0:
            errors.append(f"filter.background_threshold 需落在 (0, 1] 区间，当前为 {threshold!r}")
    except (TypeError, ValueError):
        errors.append(f"filter.background_threshold 必须是数字，当前为 {threshold!r}")

    val_ratio = cfg["dataset"].get("val_ratio")
    try:
        if not 0.0 <= float(val_ratio) < 1.0:
            errors.append(f"dataset.val_ratio 需落在 [0, 1) 区间，当前为 {val_ratio!r}")
    except (TypeError, ValueError):
        errors.append(f"dataset.val_ratio 必须是数字，当前为 {val_ratio!r}")

    return errors


def _try_cast(value, caster):
    """尝试类型转换；失败时原样返回，交由校验环节统一报错。"""
    try:
        return caster(value)
    except (TypeError, ValueError):
        return value


def normalize_config(cfg: dict) -> dict:
    """把 YAML 中可能是字符串的数值统一转成正确类型。

    转换失败不抛异常，保持原值，让 validate_config 给出一条清晰的错误提示。
    """
    cfg = copy.deepcopy(cfg)
    sl = cfg["slice"]
    sl["crop_size"] = _try_cast(sl.get("crop_size"), int)
    sl["overlap"] = _try_cast(sl.get("overlap"), float)
    sl["suffix"] = str(sl.get("suffix") or "")

    palette = sl["palette"]
    palette["field"] = str(palette.get("field") or "")
    classes = palette.get("classes")
    if isinstance(classes, list):
        palette["classes"] = [str(c) for c in classes]
    colors = palette.get("colors")
    if isinstance(colors, list):
        palette["colors"] = [
            [_try_cast(c, int) for c in color] if isinstance(color, list) else color
            for color in colors
        ]

    cfg["filter"]["background_threshold"] = _try_cast(cfg["filter"].get("background_threshold"), float)
    cfg["dataset"]["val_ratio"] = _try_cast(cfg["dataset"].get("val_ratio"), float)
    for name in STEP_ORDER:
        cfg["steps"][name] = bool(cfg["steps"].get(name))
    return cfg


# --------------------------------------------------------------------------- #
# 运行环境
# --------------------------------------------------------------------------- #
def setup_gdal_env() -> None:
    """在导入 osgeo 之前设置 GDAL / PROJ 数据目录。

    原脚本把这四行写在 `from conversion_tools... import ...` 之后，此时 osgeo
    已经完成初始化，环境变量实际上没有生效；这里提前到导入之前执行。
    """
    osgeo_env = Path(sys.prefix) / "Lib" / "site-packages" / "osgeo"
    if not osgeo_env.exists():
        return
    os.environ["GDAL_DRIVER_PATH"] = str(osgeo_env)
    os.environ["GDAL_HOME"] = str(osgeo_env)
    proj_dir = osgeo_env / "data" / "proj"
    gdal_dir = osgeo_env / "data" / "gdal"
    if proj_dir.exists():
        os.environ["PROJ_LIB"] = str(proj_dir)
    if gdal_dir.exists():
        os.environ["GDAL_DATA"] = str(gdal_dir)


def _count_files(folder: Path, suffixes: tuple[str, ...] | None = None) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir()
               if p.is_file() and (suffixes is None or p.suffix.lower() in suffixes))


# --------------------------------------------------------------------------- #
# 摘要打印
# --------------------------------------------------------------------------- #
def _pad(text: str, width: int) -> str:
    """按终端显示宽度右补空格（中日韩全角字符按 2 列计算）。"""
    import unicodedata
    shown = sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)
    return text + " " * max(0, width - shown)


def print_summary(cfg: dict, config_path: Path) -> None:
    inp, sl = cfg["input"], cfg["slice"]
    palette = sl["palette"]
    enabled = [name for name in STEP_ORDER if cfg["steps"][name]]

    print("=" * 78)
    print("样本构建配置")
    print("-" * 78)
    print(f"配置文件   : {config_path}")
    print(f"输入影像   : {inp['tif_dir']}")
    print(f"标注矢量   : {inp['shp_path']}")
    print(f"输出目录   : {inp['out_dir']}")
    print(f"裁剪设置   : crop_size={sl['crop_size']}  overlap={sl['overlap']}  suffix='{sl['suffix']}'")
    print(f"类别字段   : {palette['field']}")
    for name, color in zip(palette["classes"], palette["colors"]):
        print(f"             - {_pad(str(name), 16)} -> RGB{tuple(color)}")
    print(f"背景阈值   : {cfg['filter']['background_threshold']}")
    print(f"验证集占比 : {cfg['dataset']['val_ratio']}")
    print(f"执行步骤   : {' -> '.join(enabled)}")
    print("=" * 78)


# --------------------------------------------------------------------------- #
# 流水线
# --------------------------------------------------------------------------- #
def run_pipeline(cfg: dict, progress: bool = True) -> None:
    """按配置执行流水线。

    :param progress: 是否显示进度条（输出重定向到日志文件时可关闭）
    """
    setup_gdal_env()
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

    from conversion_tools.geo_spatial_processor import batch_slice
    from conversion_tools.make_samples import mask_dataset
    from conversion_tools.mask2txt import mask2txt
    from conversion_tools.mask_filtration import filter_samples_by_background

    inp, sl = cfg["input"], cfg["slice"]
    palette = sl["palette"]
    out_dir = Path(inp["out_dir"])

    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    images1_dir = out_dir / "images1"
    labels1_dir = out_dir / "labels1"
    txt_dir = out_dir / "txt"
    dataset_dir = out_dir / "SINGLECLASSDataset"

    started = time.time()
    report: list[tuple[str, str]] = []

    # 1. 滑窗裁剪 ---------------------------------------------------------
    if cfg["steps"]["slice"]:
        t0 = time.time()
        print(f"\n[1/4] {STEP_LABELS['slice']} ...")
        stats = batch_slice(
            inp["tif_dir"], inp["out_dir"], inp["shp_path"],
            sl["crop_size"], palette["classes"], palette["colors"], sl["suffix"],
            overlap=sl["overlap"], field=palette["field"], progress=progress,
        )
        report.append((STEP_LABELS["slice"],
                       f"影像 {stats['images']} 景  样本 {stats['saved']} 个  失败 {stats['failed']} 景"
                       f"  (images={_count_files(images_dir, ('.tif',))}"
                       f" labels={_count_files(labels_dir, ('.png',))})"
                       f"  用时 {time.time() - t0:.1f}s"))

    # 2. 样本清洗 ---------------------------------------------------------
    if cfg["steps"]["filter"]:
        t0 = time.time()
        print(f"\n[2/4] {STEP_LABELS['filter']} ...")
        filter_samples_by_background(
            str(images_dir), str(labels_dir), str(images1_dir), str(labels1_dir),
            cfg["filter"]["background_threshold"],
        )
        report.append((STEP_LABELS["filter"],
                       f"images1={_count_files(images1_dir, ('.tif',))}  "
                       f"labels1={_count_files(labels1_dir, ('.png',))}  用时 {time.time() - t0:.1f}s"))

    # 3. 掩膜转 YOLO 标签 -------------------------------------------------
    if cfg["steps"]["to_txt"]:
        t0 = time.time()
        print(f"\n[3/4] {STEP_LABELS['to_txt']} ...")
        mask2txt(str(labels1_dir), str(txt_dir), palette["classes"], palette["colors"], progress=progress)
        report.append((STEP_LABELS["to_txt"],
                       f"txt={_count_files(txt_dir, ('.txt',))}  用时 {time.time() - t0:.1f}s"))

    # 4. 划分数据集 -------------------------------------------------------
    if cfg["steps"]["dataset"]:
        t0 = time.time()
        print(f"\n[4/4] {STEP_LABELS['dataset']} ...")
        mask_dataset(inp["out_dir"], cfg["dataset"]["val_ratio"])
        train_imgs = _count_files(dataset_dir / "images" / "train")
        val_imgs = _count_files(dataset_dir / "images" / "val")
        report.append((STEP_LABELS["dataset"],
                       f"train={train_imgs}  val={val_imgs}  用时 {time.time() - t0:.1f}s"))

    # 汇总 ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("执行结果")
    print("-" * 78)
    for label, detail in report:
        print(f"{label:<22}: {detail}")
    print(f"{'总耗时':<20}: {time.time() - started:.1f}s")
    print("=" * 78)


# --------------------------------------------------------------------------- #
# 命令行入口
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="样本构建流水线（YAML 配置驱动）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python buid_mask.py -c configs/l1_1024.yaml\n"
            "  python buid_mask.py --list\n"
            "  python buid_mask.py -c configs/x.yaml --steps slice,to_txt\n"
            "  python buid_mask.py -c configs/x.yaml --set slice.crop_size=2048\n"
        ),
    )
    parser.add_argument("-c", "--config", help="YAML 配置文件路径；省略时自动使用 configs/ 下唯一的配置")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="覆盖配置项，可重复，例如 --set slice.crop_size=2048")
    parser.add_argument("--steps", help="只执行指定步骤，逗号分隔，可选：slice,filter,to_txt,dataset")
    parser.add_argument("--dry-run", action="store_true", help="仅校验配置并打印摘要，不执行流水线")
    parser.add_argument("--no-progress", action="store_true",
                        help="关闭进度条（输出重定向到日志文件时建议开启）")
    parser.add_argument("--list", action="store_true", help="列出 configs/ 下的全部配置后退出")
    return parser


def list_configs() -> None:
    if not CONFIG_DIR.is_dir():
        print(f"配置目录不存在：{CONFIG_DIR}")
        return
    items = sorted(p for p in CONFIG_DIR.iterdir()
                   if p.suffix.lower() in (".yaml", ".yml") and not p.name.startswith("_"))
    if not items:
        print(f"{CONFIG_DIR} 下暂无配置，可复制 _template.yaml 新建一份。")
        return
    print(f"可用配置（{CONFIG_DIR}）：")
    for path in items:
        print(f"  - {path.name}")
    print("\n用法：python buid_mask.py -c configs/<文件名>")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        list_configs()
        return 0

    try:
        config_path = resolve_config_path(args.config)
        cfg = load_config(config_path)

        # --steps 覆盖执行开关
        if args.steps:
            selected = {s.strip() for s in args.steps.split(",") if s.strip()}
            unknown = selected - set(STEP_ORDER)
            if unknown:
                raise ConfigError(f"--steps 中存在未知步骤：{', '.join(sorted(unknown))}，"
                                  f"可选：{', '.join(STEP_ORDER)}")
            if not selected:
                raise ConfigError("--steps 为空，未指定任何步骤")
            for name in STEP_ORDER:
                cfg["steps"][name] = name in selected

        # --set 覆盖任意配置项
        for item in args.overrides:
            if "=" not in item:
                raise ConfigError(f"--set 参数格式应为 KEY=VALUE，当前为：{item}")
            key, value = item.split("=", 1)
            deep_set(cfg, key.strip(), _parse_override_value(value.strip()))

        cfg = normalize_config(cfg)

        errors = validate_config(cfg)
        if errors:
            print("配置校验未通过：", file=sys.stderr)
            for err in errors:
                print(f"  ✗ {err}", file=sys.stderr)
            return 2

        print_summary(cfg, config_path)

        if args.dry_run:
            print("\n--dry-run 已开启，配置校验通过，未执行流水线。")
            return 0

        run_pipeline(cfg, progress=not args.no_progress)
        return 0
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
