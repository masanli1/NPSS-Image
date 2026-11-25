# Set12（标准测试集）

12 张经典灰度测试图。为遵守各图像的版权，本仓库不直接分发，
请自行下载（任选其一）：

- KAIR 仓库：`https://github.com/cszn/KAIR/tree/master/testsets/set12`
- 或搜索 "Set12 dataset" 任意镜像

放成 `data/set12/01.png ... 12.png` 即可运行：

```bash
python3 tools/finite_search.py --images data/set12
python3 benchmark.py --set12 data/set12
```
