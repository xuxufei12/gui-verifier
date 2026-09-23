# GUI Verifier dataset / 数据集

基于 [OS-Atlas](https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data) 和 [ScreenQA](https://github.com/google-research-datasets/screen_qa) 做了扩展。 / Extended from OS-Atlas and ScreenQA.

| Split | Screenshots | Judgments |
|---|---:|---:|
| train | 2251 | 10266 |
| validation | 288 | 1314 |
| calibration | 280 | 1286 |

`final/*.jsonl` 包含问题、候选、标签与固定划分；`images-manifest.jsonl` 保留逐图来源、版本、原文件名、哈希和标记框。标签为 `silver_same_model_review`，未经独立人工审核；不更改历史实验的该项事实。 / JSONL files contain questions, options, labels and fixed splits; image provenance is in the manifest. Labels are silver, not independently human-reviewed.

## Images / 图片

图片不在此仓库重复托管。请从上游获取对应图片并遵守其条款；ScreenQA 图片来自 [Rico](https://www.interactionmining.org/archive/rico)，请先阅读 [Rico terms](https://www.interactionmining.org/archive/rico/copyright.txt)。 / Images are not redistributed; acquire them from upstream under their terms.

OS-Atlas revision: `e3a4c90c5f6129c25efdaa0671b08a11f7cb8f3f`；ScreenQA revision: `1dfdbccaf56948821b5fa8ffe5d186fe4751e46d`。

本地取得原图后可重建标记截图（Arial字体需自行合法取得，不随仓库分发）：

```bash
python tools/restore_dataset_images.py --source-dir /path/to/extracted/upstream/images --font-regular /path/to/arial.ttf --font-bold /path/to/arialbd.ttf
```

脚本按上游原文件名或manifest中的id定位，核验原图hash后渲染，报告缺图或结果hash不一致；不会下载、自动接受上游条款或覆盖不同图片。字体/Pillow版本会影响重建hash，原实验Pillow为12.2.0。训练时将 `--train` 等参数指向此目录的 `final/*.jsonl`。 / Reconstruction requires local upstream images and fonts; mismatches are reported, never silently treated as exact reproduction.

## License / 许可

新增标注按 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 提供，署名 GUI Verifier contributors；保留OS-Atlas作者与ScreenQA作者的来源归属。上游材料分别按原许可使用，详见 `licenses/`；根目录MIT只适用于代码。原图不因标注许可而获得再授权。 / Added annotations: CC BY 4.0, attribution to GUI Verifier contributors and the upstream OS-Atlas and ScreenQA authors. Upstream terms continue to apply; the code MIT license does not relicense images.
