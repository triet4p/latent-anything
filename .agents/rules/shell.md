# Shell Command Execution Rules

## Commands that require the external environment

Shell hoặc PowerShell commands sử dụng các công cụ hệ thống hoặc công cụ có thể
truy cập tài nguyên bên ngoài sandbox **phải được chạy trong môi trường bên
ngoài sandbox** (`sandbox_permissions: require_escalated`). Điều này áp dụng cho
các lệnh như:

- `docker` và `docker compose`.
- `openssl`.
- `gh` (GitHub CLI).
- `git`.

Quy tắc này cũng áp dụng cho các command tương tự nếu chúng cần network,
credential, daemon, socket, hoặc tài nguyên hệ thống mà sandbox có thể hạn chế.

Không được coi một command là đã chạy thành công chỉ vì nó chạy được trong
sandbox khi command đó thuộc nhóm trên. Nếu command thất bại do sandbox hoặc
có dấu hiệu bị sandbox hạn chế, hãy chạy lại với `sandbox_permissions:
require_escalated` và kèm lý do xin quyền ngắn gọn cho người dùng.
