# Data Contract

## 1. Data Unit

Each row = one earnings call event

Primary key:
- event_id

---

## 2. Required Fields

### Target

- shock_minus_pre

---

### Market Features

- pre_call_volatility
- returns
- volume

---

### Controls

- firm_size
- sector
- historical volatility

---

### ECC Features

- text_embedding
- qa_embedding
- audio_features

---

### Proxy Features

- transcript_coverage
- alignment_score
- audio_completeness

---

## 3. Split Columns

- train_flag
- val_flag
- test_flag

---

## 4. Missing Values

- drop if critical missing
- otherwise impute (mean or median)

---

## 5. Constraints

- NO future information
- All features must be available at prediction time