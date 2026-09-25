"""python -m clew_ml.preprocess — align + cache every IO-VNBD drive to data/processed/*.npz"""
from .data import DT, drive_folders, load_drive, save, split_of


def main():
    kept = skipped = 0
    hours = {"train": 0.0, "val": 0.0, "test": 0.0}
    for f in drive_folders():
        segs = load_drive(f)
        if not segs:
            skipped += 1
            continue
        for d in segs:
            save(d)
            kept += 1
            hours[split_of(d.name)] += d.T * DT / 3600
            print(f"ok    {d.name:45s} {d.T * DT / 60:6.1f} min  lag {d.lag * DT:+7.1f} s  imu {d.imu_lag * DT:+5.1f} s r={d.imu_corr:.2f}  split {split_of(d.name)}")
    print(f"segments {kept}, recordings without a usable segment {skipped}; hours " + ", ".join(f"{k} {v:.1f}" for k, v in hours.items()))


if __name__ == "__main__":
    main()
