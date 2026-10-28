#!/usr/bin/env python3
"""
check_timing.py — Timing-Analyse einer Teach-Aufnahme.
 
Prüft für eine Aufnahme:
  1. pro Topic: Rate, Gleichmäßigkeit (Jitter) der Header-Zeitstempel und
     Aufnahme-Latenz (Empfangszeit minus Header-Zeitstempel).
  2. Synchronität der Streams: Versatz von Kamera-Topics gegenüber
     /joint_states (nearest-neighbour der Header-Zeitstempel).
  3. Zuordnung waypoints.jsonl -> rosbag: für jedes Waypoint-t das nächste
     /joint_states-Sample und die Zeitlücke dazu.
 
Aufruf:
    python3 check_timing.py /workspace/cobot_recordings/<zeitstempel>/teach
 
(Es wird der Unterordner rosbag/ und die Datei waypoints.jsonl erwartet.)
"""
 
import glob
import json
import os
import statistics
import sys
 
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
 
REF_TOPIC = "/joint_states"
 
 
def find_bag_dir(path):
    """Akzeptiert teach-Ordner oder direkt den rosbag-Ordner."""
    if glob.glob(os.path.join(path, "*.db3")) or glob.glob(os.path.join(path, "*.mcap")):
        return path
    cand = os.path.join(path, "rosbag")
    if os.path.isdir(cand):
        return cand
    raise SystemExit(f"Kein rosbag gefunden unter {path} (weder *.db3/*.mcap noch rosbag/).")
 
 
def storage_id(bag_dir):
    if glob.glob(os.path.join(bag_dir, "*.mcap")):
        return "mcap"
    return "sqlite3"
 
 
def get_stamp(msg):
    """Header-Zeitstempel in Sekunden, oder None.
 
    Ein Zeitstempel von 0 (sec==0 und nanosec==0) gilt als 'nicht gesetzt'
    -- das betrifft z. B. statische Transforms (/tf_static) und wird als
    None behandelt, damit es die Statistik nicht verfälscht.
    """
    s = None
    if hasattr(msg, "header"):
        s = msg.header.stamp
    elif hasattr(msg, "transforms") and msg.transforms:
        s = msg.transforms[0].header.stamp
    if s is None:
        return None
    if s.sec == 0 and s.nanosec == 0:
        return None
    return s.sec + s.nanosec * 1e-9
 
 
def read_bag(bag_dir):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_dir, storage_id=storage_id(bag_dir)),
        rosbag2_py.ConverterOptions("", ""),
    )
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    msgclass = {}
    # per topic: lists of recv-time (s) and header-stamp (s)
    data = {name: {"recv": [], "stamp": []} for name in type_map}
 
    while reader.has_next():
        topic, raw, t_recv = reader.read_next()
        if topic not in msgclass:
            try:
                msgclass[topic] = get_message(type_map[topic])
            except Exception:
                msgclass[topic] = None
        cls = msgclass[topic]
        if cls is None:
            continue
        try:
            msg = deserialize_message(raw, cls)
        except Exception:
            continue
        data[topic]["recv"].append(t_recv * 1e-9)
        st = get_stamp(msg)
        data[topic]["stamp"].append(st)
    return data, type_map
 
 
def intervals(seq):
    return [b - a for a, b in zip(seq, seq[1:])]
 
 
def fmt_ms(x):
    return f"{x * 1000:8.2f} ms"
 
 
def analyse_topics(data, type_map):
    print("=" * 74)
    print("1) PRO TOPIC — Rate, Jitter (Header), Aufnahme-Latenz (recv - header)")
    print("=" * 74)
    for topic in sorted(data):
        recv = data[topic]["recv"]
        stamp = [s for s in data[topic]["stamp"] if s is not None]
        n = len(recv)
        if n < 2:
            print(f"\n{topic}  ({type_map[topic]})\n  nur {n} Nachricht(en) — übersprungen")
            continue
        dur = recv[-1] - recv[0]
        rate = (n - 1) / dur if dur > 0 else float("nan")
        print(f"\n{topic}  ({type_map[topic]})")
        print(f"  Nachrichten: {n}   Dauer: {dur:6.2f} s   Rate: {rate:6.2f} Hz")
        if len(stamp) >= 2:
            di = intervals(stamp)
            print(f"  Header-Intervall:  Median {fmt_ms(statistics.median(di))}"
                  f"  min {fmt_ms(min(di))}  max {fmt_ms(max(di))}"
                  f"  std {fmt_ms(statistics.pstdev(di))}")
            # Aufnahme-Latenz: nur wo beide vorhanden
            lat = [r - s for r, s in zip(recv, data[topic]["stamp"]) if s is not None]
            if lat:
                print(f"  Aufnahme-Latenz (recv-header): Median {fmt_ms(statistics.median(lat))}"
                      f"  min {fmt_ms(min(lat))}  max {fmt_ms(max(lat))}")
        else:
            print("  (kein Header-Zeitstempel in diesem Topic)")
 
 
def nearest_offset(ref_stamps, other_stamps):
    """Median des Betrags des nächsten Nachbarn (other - nächster ref)."""
    if not ref_stamps or not other_stamps:
        return None
    ref = sorted(ref_stamps)
    import bisect
    offs = []
    for s in other_stamps:
        i = bisect.bisect_left(ref, s)
        cands = []
        if i < len(ref):
            cands.append(ref[i])
        if i > 0:
            cands.append(ref[i - 1])
        nearest = min(cands, key=lambda r: abs(r - s))
        offs.append(s - nearest)
    return offs
 
 
def analyse_sync(data):
    print("\n" + "=" * 74)
    print("2) SYNCHRONITÄT — Versatz anderer Streams gegenüber /joint_states")
    print("=" * 74)
    ref = [s for s in data.get(REF_TOPIC, {}).get("stamp", []) if s is not None]
    if not ref:
        print(f"  {REF_TOPIC} nicht im Bag — übersprungen.")
        return
    for topic in sorted(data):
        if topic == REF_TOPIC:
            continue
        other = [s for s in data[topic]["stamp"] if s is not None]
        offs = nearest_offset(ref, other)
        if not offs:
            continue
        med = statistics.median(offs)
        absmed = statistics.median([abs(o) for o in offs])
        print(f"\n{topic}")
        print(f"  Versatz zu {REF_TOPIC}: Median {fmt_ms(med)}"
              f"  (Betrag-Median {fmt_ms(absmed)},"
              f"  min {fmt_ms(min(offs))}  max {fmt_ms(max(offs))})")
 
 
def analyse_waypoints(data, teach_dir):
    print("\n" + "=" * 74)
    print("3) WAYPOINTS ↔ ROSBAG — passt jedes Waypoint-t zu einem Sample?")
    print("=" * 74)
    wp_path = os.path.join(teach_dir, "waypoints.jsonl")
    if not os.path.isfile(wp_path):
        print(f"  {wp_path} nicht gefunden — übersprungen.")
        return
    ref = sorted(s for s in data.get(REF_TOPIC, {}).get("stamp", []) if s is not None)
    if not ref:
        print(f"  {REF_TOPIC} nicht im Bag — übersprungen.")
        return
    import bisect
    gaps = []
    with open(wp_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            wp = json.loads(line)
            t = wp.get("t", 0.0)
            if not t:
                continue  # init/last mit t=0 überspringen
            i = bisect.bisect_left(ref, t)
            cands = []
            if i < len(ref):
                cands.append(ref[i])
            if i > 0:
                cands.append(ref[i - 1])
            nearest = min(cands, key=lambda r: abs(r - t))
            gaps.append(abs(nearest - t))
    if not gaps:
        print("  Keine Waypoints mit gültigem t gefunden.")
        return
    print(f"  Waypoints mit t: {len(gaps)}")
    print(f"  Abstand zum nächsten /joint_states-Sample:"
          f"  Median {fmt_ms(statistics.median(gaps))}"
          f"  max {fmt_ms(max(gaps))}")
    print("  (Sollte << Sample-Intervall von 0,1 s sein — dann passt die Zuordnung.)")
 
 
def main():
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)
    teach_dir = sys.argv[1].rstrip("/")
    bag_dir = find_bag_dir(teach_dir)
    print(f"Bag: {bag_dir}")
    data, type_map = read_bag(bag_dir)
    analyse_topics(data, type_map)
    analyse_sync(data)
    analyse_waypoints(data, teach_dir)
    print("\nFertig.")
 
 
if __name__ == "__main__":
    main()
