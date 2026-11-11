"""Explicit frozen-profile entry for the existing observation service."""
from .store import ProfileStore
from .schema import compile_config


def load_locked_config(root='/sdcard/transport_mvp/tuning_profiles',slot=None):
    store=ProfileStore(root)
    slot=store.selected_slot() if slot is None else slot
    record=store.load(slot)
    if record is None:raise ValueError('No valid saved tuning profile')
    profile=record['profile']
    if profile['locked'] is not True:raise ValueError('Saved profile must be explicitly locked')
    config=compile_config(profile)
    config.update(TUNING_SLOT=slot,TUNING_GENERATION=record['generation'],TUNING_SHA256=record['sha256'],
        DEFAULT_MODE='QR',DEFAULT_TARGET_ID=None,DEFAULT_STATION=None)
    return config
