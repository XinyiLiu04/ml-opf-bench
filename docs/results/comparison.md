## Leaderboard

| Method                | MAE | Violations | Inference(ms) | Rate(%) |
|---------------------|-------------|----------|--------------|---------------|
| Ipopt       | 0.2±0.1     | 125.3±5.2 | 500±30       | 100.0         |
| DNN   | 0.5±0.3     | 130.1±6.8 | 300±45       | 98.5          |
| PINN            | 0.3±0.2     | 127.5±4.9 | 10±2         | 99.2          |
| RL   | 0.22±0.12   | 125.8±3.5 | 5±1          | 99.9          |

## Visualization
![Inference](assets/images/speed_comparison.png)
![Accuracy-Speed tradeoff](assets/images/accuracy_speed.png)