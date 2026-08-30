"""
blinker signals fired by canela's detectors.

Every signal sends the detector instance itself as `sender` (so a receiver
can reach `sender.cfg`, `sender.label`, etc.). Connect a receiver anywhere
without touching detector code:

    from canela.signals import event_saved

    def my_receiver(sender, **kwargs):
        ...

    event_saved.connect(my_receiver)
"""

from blinker import Namespace

canela_signals = Namespace()

# A detector thread starting/stopping its run() loop.
detector_started = canela_signals.signal("detector-started")
detector_stopped = canela_signals.signal("detector-stopped")

# RTSP / ffmpeg source connectivity.
stream_connected = canela_signals.signal("stream-connected")
stream_disconnected = canela_signals.signal("stream-disconnected")

# Fires on every raw detection, regardless of cooldown gating.
# kwargs: result (MotionResult | AudioResult)
detection_event = canela_signals.signal("detection-event")

# Fires only when an event actually clears the cooldown and gets written to disk.
# kwargs: result (MotionResult | AudioResult), timestamp (str)
event_saved = canela_signals.signal("event-saved")