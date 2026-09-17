# Sprint 79 OpenVLA scope revision

The supported release-evidence ceiling is 16 GiB VRAM. Canonical BF16 `THY-X01-OPENVLA` / M14 L19 requires a defensible >=24 GiB CUDA host, so it is hardware-excluded from active Sprint 79 execution, M14 applicable-lane gates, and theory-coverage denominators.

The historical D0 configs and feasibility receipts remain retained and discoverable: [`l19-openvla.config.json`](m14/l19-openvla.config.json), [`l19-openvla.json`](m14/l19-openvla.json), and [`l19-openvla-16gb-feasibility.json`](m14/l19-openvla-16gb-feasibility.json). They are not a pass, waiver, or support claim. Quantized/offloaded execution would be a different contract and was not substituted. Reintroducing L19 requires an explicit scope revision, adequate hardware, the reviewed adapter, and the declared 500-trial intervention contract.

The active map/queue contain 39 rows; the excluded map record and queue reconciliation retain the 40-row historical inventory without positional corruption. Coverage is 41/63 core and 41/64 scoped overall; thresholds are 60 and 58, leaving shortfalls of 19 and 17. M14 remains 24 historical rows: 23 applicable (13 accepted, 2 partial, 1 pending, 7 blocked) plus 1 hardware-excluded L19. L21 SmolVLA remains active at its documented ~16 GiB profile. No empirical gate, release tag, version, or OpenVLA claim changed.
