## WSL 与 WSL2 概念解析

- **WSL（Windows Subsystem for Linux）**：一个 Windows 特性，允许在 Windows 上运行 Linux 二进制文件（ELF）的兼容层 
- **WSL1**：兼容层方式执行 Linux 用户空间程序，并直接使用 Windows 文件系统，无虚拟化；
- **WSL2**：基于轻量级 Hyper‑V VM，使用真正的 Linux 内核，I/O 性能提升显著（读写速度为 WSL1 的约 20 倍）

------



## WSL2 + Ubuntu 环境搭建

1. **确认系统版本**

   - 需 Windows 10 ≥19041（2004）或 Windows 11

2. **启用 WSL 与虚拟机平台**

   - 以管理员权限运行 PowerShell：

   ```
   wsl --install
   ```

   此命令自动开启 WSL、VM 平台，安装 Linux kernel 并设置 WSL2 默认，默认部署 Ubuntu 

   手动安装路径（旧版 Windows）：使用 `dism.exe /enable-feature /featurename:Microsoft‑Windows‑Subsystem‑for‑Linux /all /norestart` 然后启用 `VirtualMachinePlatform` 

3. **设置 WSL 版本**

   - 查看版本：`wsl -l -v`
   - 设置为 WSL2：`wsl --set-version Ubuntu-20.04 2` 或全局：`wsl --set-default-version 2`

4. **首次启动 Ubuntu**

   - 第一次打开会解压安装并提示创建 Linux 用户名与密码

5. **常用 WSL 命令**

   - 创建、列出、设默认发行版：`wsl --install -d <Distro>`、`wsl -l -v`、`wsl -s <Distro>` 
   - Windows 与 Linux 互操作：如 `wsl ls -la` 或 `wsl.exe pwd`；跨环境混合命令 `wsl ls -la \| findstr git`

------



## Linux 基本操作

以下是学习到的基础命令与操作技巧：

```
- **文件与目录管理**
  - `ls`, `cd`, `pwd`
  - `mkdir`, `rm`, `cp`, `mv`

- **文本查看与编辑**
  - `cat`, `less`, `head`, `tail`
  - 编辑器：`vi`, `nano`, `vim` :contentReference[oaicite:29]{index=29}

- **权限与进程**
  - 权限查看：`ls -l`, 修改权限：`chmod`, `chown` :contentReference[oaicite:30]{index=30}
  - 进程管理：`ps`, `top`, `kill`, `bg`, `fg`, `jobs` :contentReference[oaicite:31]{index=31}

- **软件包管理（以 Ubuntu 为例）**
  - `sudo apt update`, `sudo apt install <软件包>`
  - 其他命令：`apt remove`, `apt upgrade` :contentReference[oaicite:32]{index=32}

- **Shell 脚本基础**
  - 编写 `.sh` 文件，加入 `#!/bin/bash` 声明；
  - 使用变量、条件语句、循环；
  - 示例：`myscript.sh`，执行 `bash myscript.sh`

- **管道与重定向**
  - 管道 `|`: 如 `ls -la | grep txt`
  - 输出重定向 `>` 与 `>>`
  - 输入重定向 `<`

- **环境变量与别名**
  - 查看：`echo $PATH`, 设置：`export PATH=$PATH:/usr/local/bin`
  - 临时别名：`alias ll='ls -la'`

- **远程操作**
  - `ssh user@host`
  - 文件传输：`scp`, `rsync`
```

## 