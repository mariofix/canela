"""
Built-in signal receivers. Wired up in cli.main(); add more the same way -
connect a function to any signal in canela.signals, no detector changes needed.
"""

import json
import os

from .signals import event_saved


def log_event_jsonl(sender, **kwargs):
    """Append a one-line JSON audit record for every saved event, into
    that stream's own save_dir. Cheap correlation trail across streams/servers."""
    record = {
        "stream": sender.cfg.name,
        "detector": sender.label,
        **{k: v for k, v in kwargs.items() if k != "result"},
    }

    path = os.path.join(sender.cfg.save_dir, "events.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")