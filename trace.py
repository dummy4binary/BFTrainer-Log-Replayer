import numpy as np
import pandas as pd

def deduplicate_events(ndstat):
    state_map = ndstat.values.copy()
    dedup_map = [state_map[0], ]
    evt_ts = [ndstat.index[0], ]
    pstat = state_map[0]
    for i in range(1, state_map.shape[0]):
        if np.abs(state_map[i] - pstat).sum() > 0:
            dedup_map.append(state_map[i])
            evt_ts.append(ndstat.index[i])
            pstat = state_map[i]
    return pd.DataFrame(dedup_map, columns=ndstat.columns, index=evt_ts)