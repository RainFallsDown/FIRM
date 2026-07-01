#!/bin/bash
set -e
echo "==========================================" 
echo "重新启动Tianqing BC训练"
echo "==========================================" 
cd ${DREAMZERO_ROOT:-/path/to/dreamzero}/code/dreamzero
echo ""
echo "1. 检查视频切分结果..."
VIDEO_COUNT=$(find ${TIANQING_DATA_ROOT:-/path/to/tianqing}/tianqing_data/data_valid/A2p_dataset_0302_330/videos -name "*.mp4" 2>/dev/null | wc -l)
echo "   视频文件总数: $VIDEO_COUNT"
echo "   预期: 990 (330 episodes × 3 cameras)"
echo ""
echo "2. 检查现有训练进程..."
EXISTING_PIDS=$(ps aux | grep "torchrun.*train_wam.py" | grep -v grep | awk '{print $2}')
if [ -n "$EXISTING_PIDS" ]; then
    echo "   发现现有训练进程: $EXISTING_PIDS"
    echo "   正在停止..."
    kill $EXISTING_PIDS 2>/dev/null || true
    sleep 3
else
    echo "   ✓ 没有现有训练进程"
fi
echo ""
echo "3. 检查GPU状态..."
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{printf "   GPU %s: 使用率 %s%% | 显存 %s/%s MB\n", $1, $2, $3, $4}'
echo ""
echo "4. 启动BC训练..."
bash scripts/train/start_tianqing_training.sh
sleep 10
echo ""
echo "5. 训练进程状态:"
ps aux | grep "torchrun.*train_wam.py" | grep -v grep | awk '{printf "   PID: %s | CPU: %s%% | MEM: %s%%\n", $2, $3, $4}'
LATEST_LOG=$(ls -t logs/tianqing_training_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo ""
    echo "6. 日志文件: $LATEST_LOG"
    echo "   最后20行:"
    tail -20 "$LATEST_LOG" | sed 's/^/   /'
fi
echo ""
echo "==========================================" 
echo "训练已启动！"
echo "==========================================" 
