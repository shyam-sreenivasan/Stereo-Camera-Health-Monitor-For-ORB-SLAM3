import csv


def convert_gt_to_tum(gt_csv_path, output_path):
    """TUM-VI mocap CSV -> TUM format (t tx ty tz qx qy qz qw)."""
    if output_path.exists():
        return
    with open(gt_csv_path) as f, open(output_path, "w") as out:
        for row in csv.reader(f):
            if row[0].startswith("#"):
                continue
            t = float(row[0]) / 1e9
            px, py, pz = row[1], row[2], row[3]
            qw, qx, qy, qz = row[4], row[5], row[6], row[7]
            out.write(f"{t:.9f} {px} {py} {pz} {qx} {qy} {qz} {qw}\n")


def convert_orbslam_to_tum(orb_path, tum_path):
    """ORB-SLAM3 EuRoC output -> TUM format."""
    with open(orb_path) as f, open(tum_path, "w") as out:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            t = float(parts[0])
            if t > 1e15:
                t /= 1e9
            out.write(
                f"{t:.9f} {parts[1]} {parts[2]} {parts[3]} "
                f"{parts[4]} {parts[5]} {parts[6]} {parts[7]}\n"
            )
