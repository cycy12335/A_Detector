# External CSV batch workflow

The Docker demo keeps the scoring service strict: it accepts JSON feature
records, not CSV uploads. The optional `csv-runner` Compose service provides the
CSV adapter around that service.

## Host-side usage

```powershell
python scripts/score_external_csv.py `
  --input data/incoming/external.csv `
  --output data/output/external_scores.csv `
  --id-column TransactionID `
  --label-column isFraud
```

先做小样本冒烟测试时可以加 `--max-rows 100`，确认接口、字段和输出都正常后再去掉它。

The adapter reads `/v1/model` first and uses its `required_features` list as the
authority. It fails before scoring when a required feature is missing, ignores
extra CSV columns, converts missing values to JSON `null`, and sends batches of
at most 100 rows. `isFraud` is retained only in the local output for optional
evaluation; it is never sent to the scoring service.

## Docker usage

Put the CSV at `data/incoming/external.csv`, start the normal demo, then run the
profile-gated adapter:

```powershell
docker compose -f compose.demo.yaml up --build -d risk-demo
docker compose -f compose.demo.yaml --profile batch run --rm csv-runner
```

Results are written to:

```text
data/output/external_scores.csv
data/output/external_scores.csv.meta.json
```

The adapter does not fit or recompute model preprocessing. The accepted bundle
performs the frozen training-time imputation, scaling, and categorical encoding.
An incomplete run is left with a `.partial` suffix and is never renamed as a
completed result.
