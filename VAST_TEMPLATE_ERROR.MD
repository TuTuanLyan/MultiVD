# Báo lỗi template Vast — image `PyTorch` (vast-ai/base-image, derivative `pytorch`)

Đo trên 2 instance ngày 2026-08-27 (RTX 5060 Ti, RTX 5070 Ti). Cả hai giống hệt nhau.

---

## Lỗi 1 — Bộ thư viện ML bị chẻ đôi giữa hai môi trường Python, không môi trường nào chạy được một mình

| Gói | `python3` hệ thống (3.12.3) | `/venv/main` (3.12.13) |
|---|---|---|
| **torch** | **THIẾU** | 2.11.0+cu128 |
| **transformers** | 4.57.1 | **THIẾU** |
| **scikit-learn** | 1.7.2 | **THIẾU** |
| **pandas** | 2.3.3 | **THIẾU** |
| **scipy** | 1.18.1 | **THIẾU** |
| numpy | 2.2.6 | 2.5.2 |

Một job `transformers` + `torch` không chạy được bằng bất kỳ môi trường nào trong hai môi trường trên, nếu không tự cài thêm.

Kiểu hỏng này tệ hơn "thiếu hẳn": `import transformers` thành công nên script chạy được một đoạn rồi mới chết ở `import torch` — trông y hệt như image không có torch.

```
$ python3 -c "import torch"
ModuleNotFoundError: No module named 'torch'

$ /venv/main/bin/python -c "import transformers"
ModuleNotFoundError: No module named 'transformers'
```

**Đề xuất:** cài đủ `transformers / scikit-learn / pandas / scipy` vào `/venv/main` (môi trường mà chính `/etc/vast-agents-guide.md` chỉ định là mặc định). Hoặc ngược lại, bỏ hẳn transformers/sklearn/pandas khỏi python hệ thống — một môi trường nửa vời nguy hiểm hơn một môi trường trống.

---

## Lỗi 2 — `/etc/environment` trỏ vào thư mục không tồn tại; không có đường nào tự động tới `/venv/main`

```
$ cat /etc/environment
PATH="/opt/instance-tools/bin:/opt/sys-venv/shim:/usr/local/nvidia/bin:..."
                              ^^^^^^^^^^^^^^^^^^
$ ls /opt/sys-venv
ls: cannot access '/opt/sys-venv': No such file or directory
```

`/opt/sys-venv` **không tồn tại trên filesystem**. Nếu shim này là cơ chế dự kiến để `python3` trỏ về venv thì nó đang không có.

PATH mà phiên SSH thực nhận cũng không khớp `/etc/environment` — thiếu cả `/opt/sys-venv/shim` lẫn `/usr/local/nvidia/bin`:

```
$ ssh host 'echo $PATH'
/opt/instance-tools/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

$ ssh host 'bash -lc "command -v python3"'
/usr/bin/python3           # kể cả LOGIN shell cũng không tới được venv

$ ssh host 'echo ACTIVE_VENV=$ACTIVE_VENV'
ACTIVE_VENV=               # không được đặt
```

Hệ quả: cách duy nhất tới được môi trường đúng là `source /venv/main/bin/activate` **thủ công**. Mọi automation gọi kiểu `ssh host 'python3 train.py'` — cách chuẩn để lái một máy thuê — đều rơi im lặng vào môi trường thiếu torch.

**Đề xuất:** tạo thật `/opt/sys-venv/shim` với `python`/`python3` trỏ về `/venv/main/bin/`, **hoặc** đặt `ACTIVE_VENV=main` và cho `/venv/main/bin` vào đầu PATH ở tầng Docker `ENV` (sshd kế thừa được), thay vì chỉ ghi trong `/etc/environment`.

---

## Lỗi 3 — `sshd_config` khai báo `UsePAM` hai lần, ngược nhau

```
$ grep -n "^ *UsePAM" /etc/ssh/sshd_config
94:UsePAM yes
134:UsePAM no
```

OpenSSH lấy lần khai báo đầu tiên nên thực tế là `yes`, nhưng đây là hai lớp cấu hình chồng lên nhau ngoài ý muốn. Liên quan trực tiếp tới Lỗi 2 vì `pam_env` là thứ nạp `/etc/environment`.

---

## Ghi chú thêm (không phải lỗi template)

**SSH key không gắn sẵn.** Tài khoản không có key nào; và không tạo được ở cấp team:

```
$ vastai show ssh-keys
[]
$ vastai create ssh-key "<pubkey>"
Failed with error 400: Team SSH keys are not supported.
SSH keys can only be created in personal context.
```

Phải `vastai attach ssh <instance_id> "<pubkey>"` cho **từng** instance. Nếu nhóm dùng team account thì cần quy trình gắn theo instance hoặc gắn qua onstart script.

**Đĩa container 20 GB.** `/` là overlay 20 GB, base image đã chiếm ~8 GB → còn ~11 GB. Nếu template hướng tới huấn luyện thì nên nâng disk mặc định, hoặc ghi rõ trong README là kho model phải để trên volume gắn ngoài.

---

## Những thứ template làm ĐÚNG

`git`, `rsync`, `curl`, `flock`, `uv`, `nvidia-smi` đều có sẵn. torch khớp đúng `cu128` cho Blackwell `sm_120`, CUDA 12.8 hoạt động thật trên cả RTX 5060 Ti lẫn 5070 Ti — chỗ này thường hay sai với GPU đời mới, template làm đúng. Tài liệu `/etc/vast-agents-guide.md` có ghi rõ `source /venv/main/bin/activate`; vấn đề nằm ở chỗ không có đường nào khác dẫn tới đó, chứ không phải tài liệu thiếu.

---

## Một dòng tóm tắt

> Bộ ML bị chẻ đôi giữa `python3` hệ thống (có transformers, **không** torch) và `/venv/main` (có torch, **không** transformers), trong khi `/opt/sys-venv/shim` khai trong PATH lại **không tồn tại** và `ACTIVE_VENV` không được đặt — nên automation qua SSH luôn rơi vào môi trường thiếu.
