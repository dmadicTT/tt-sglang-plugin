"""Tenstorrent device identity for the SGLang platform plugin."""

from sglang.srt.platforms.device_mixin import DeviceMixin, PlatformEnum


class TenstorrentDeviceMixin(DeviceMixin):
    _enum = PlatformEnum.OOT
    device_name = "tenstorrent"
    device_type = "tt"
