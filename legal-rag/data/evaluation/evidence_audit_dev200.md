# Evidence Audit Report (dev200 - Class D)

## 1. Class D Root Cause Distribution

| Root Cause Class | Description | Count | Percentage |
|---|---|---|---|
| **D1** | Article đúng nhưng sai Điều | 0 | 0.0% |
| **D2** | Đúng Điều nhưng thiếu Khoản | 2 | 10.0% |
| **D3** | Đúng Khoản nhưng thiếu Điểm/đoạn | 3 | 15.0% |
| **D4** | Child evidence retrieved nhưng bị Context Builder lọc (max doc limit) | 1 | 5.0% |
| **D5** | Evidence ở neighbor/parent nhưng expansion không lấy | 2 | 10.0% |
| **D6** | Evidence bị context budget cắt | 0 | 0.0% |
| **D7** | Evidence không tồn tại trong candidate pool (missed by retrieval) | 12 | 60.0% |
| **D8** | Parser/chunking làm mất cấu trúc evidence | 0 | 0.0% |
| **D9** | Khác | 0 | 0.0% |

## 2. Detailed Forensic Audit Table (20 Class D Cases)

| QID | Gold Article | Gold Khoản | Context Has Article? | Context Has Khoản? | Lifecycle Stage Missing | Root Cause | Explanation |
|---|---|---|---|---|---|---|---|
| 48671 | 55 | 1 | Yes | No | Retrieval | **D7** | Mismatched gold article clauses: missing ['1']. |
| 145143 | 1, 2, 44, 86, 87 | 3 | Yes | No | Retrieval | **D7** | Mismatched gold article clauses: missing ['3']. |
| 83405 | 2 | 1 | Yes | No | Context Construction | **D2** | Mismatched gold article clauses: missing ['1']. |
| 27167 | 6 | 3 | Yes | No | Context Construction | **D2** | Mismatched gold article clauses: missing ['3']. |
| 116111 | 34 | 1, 3 | Yes | No | Context Construction | **D3** | Mismatched gold article clauses: missing ['1', '3']. |
| 155465 | 3, 108 | 1, 3 | Yes | No | Retrieval | **D7** | Mismatched gold article clauses: missing ['1', '3']. |
| 16727 | 103 | 1 | Yes | No | Context Construction | **D3** | Mismatched gold article clauses: missing ['1']. |
| 7731 | 3, 168, 194 | 1, 21 | Yes | No | Retrieval | **D7** | Mismatched gold article clauses: missing ['1', '21']. |
| 104867 | 10 | 3 | Yes | No | Retrieval | **D7** | Mismatched gold article clauses: missing ['3']. |
| 21605 | 2, 15 | 3, 4, 5, 6, 7, 18 | Yes | No | Retrieval | **D5** | Mismatched gold article clauses: missing ['3', '4', '5', '6', '7', '18']. |
| 4475 | 1, 24, 33, 61 | 1, 3, 11 | Yes | No | Retrieval | **D7** | Mismatched gold article clauses: missing ['11']. |
| 107175 | 1, 79, 188 | 1, 2, 3, 4, 38 | Yes | No | Retrieval | **D7** | Mismatched gold article clauses: missing ['38']. |
| 56437 | 3 | 2, 3 | Yes | No | Retrieval | **D7** | Mismatched gold article clauses: missing ['2', '3']. |
| 167645 | 7, 75, 77, 159 | 1, 2, 3 | Yes | No | Retrieval | **D5** | Mismatched gold article clauses: missing ['3']. |
| 122267 | 36 | 1 | Yes | No | Retrieval | **D7** | Mismatched gold article clauses: missing ['1']. |
| 135981 | 7 | 1 | Yes | No | Retrieval | **D7** | Mismatched gold article clauses: missing ['1']. |
| 163025 | 33, 43 | 3 | Yes | No | Retrieval | **D7** | Mismatched gold article clauses: missing ['3']. |
| 28895 | 4 | 5 | Yes | No | Context Construction | **D3** | Mismatched gold article clauses: missing ['5']. |
| 104719 | 5, 7, 8 | 3 | Yes | No | Context Construction | **D4** | Mismatched gold article clauses: missing ['3']. |
| 99461 | 2, 3, 6 | 1 | Yes | No | Retrieval | **D7** | Mismatched gold article clauses: missing ['1']. |