"""Build a local Qlib snapshot from Tushare bars and dated CSI 300 intervals."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from threading import Lock, local
from time import monotonic, sleep

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .qlib_dump_bin import DumpDataAll


START = '20080601'
ANCHOR = '20230115'
PRICES = ['open', 'high', 'low', 'close']
# SSE 2010 annual report confirms that 600849 changed its code to 601607.
OLD_CODES = {'600849.SH': ('601607.SH', '20100305')}
FIELDS = {
    'daily': ['ts_code', 'trade_date', *PRICES, 'vol', 'amount'],
    'adj_factor': ['ts_code', 'trade_date', 'adj_factor'],
    'trade_cal': ['cal_date', 'is_open'],
    'index_weight': ['con_code', 'trade_date'],
}


class Tushare:
    def __init__(self, token, requests_per_minute):
        self.token = token
        self.interval = 60 / requests_per_minute
        self.lock = Lock()
        self.next_request = 0.0
        self.thread = local()

    def _session(self):
        if hasattr(self.thread, 'session'):
            return self.thread.session
        session = requests.Session()
        retry = Retry(total=3, status_forcelist=(429, 502, 503, 504),
                      allowed_methods={'POST'}, backoff_factor=1)
        session.mount('https://', HTTPAdapter(max_retries=retry))
        self.thread.session = session
        return session

    def query(self, name, **params):
        payload = {'api_name': name, 'token': self.token, 'params': params,
                   'fields': ','.join(FIELDS[name])}
        with self.lock:
            sleep(max(0, self.next_request - monotonic()))
            self.next_request = monotonic() + self.interval
        if os.name == 'nt':
            # Windows curl uses the system TLS and proxy configuration. Keep the token on stdin.
            response = subprocess.run(
                ['curl.exe', '--fail', '--silent', '--show-error', '--max-time', '30',
                 '--retry', '2', '--header', 'Content-Type: application/json',
                 '--data-binary', '@-', 'https://api.tushare.pro'],
                input=json.dumps(payload).encode('utf-8'), stdout=subprocess.PIPE, check=True,
            )
            result = json.loads(response.stdout)
        else:
            response = self._session().post('https://api.tushare.pro', timeout=30,
                                            allow_redirects=False, json=payload)
            response.raise_for_status()
            result = response.json()
        if result['code'] != 0:
            message = str(result['msg']).replace(self.token, '[redacted]')
            raise RuntimeError(f'Tushare {name} ({result["code"]}): {message}')
        data = result['data']
        return pd.DataFrame(data['items'], columns=data['fields']).astype(
            {column: str for column in ('trade_date', 'cal_date') if column in data['fields']}
        )


def probe(api):
    for date in ('20090105', '20190701', '20211231'):
        for name in ('daily', 'adj_factor'):
            data = api.query(name, ts_code='600000.SH', trade_date=date)
            if len(data) != 1:
                raise ValueError(f'Tushare {name} did not return the expected bar for {date}')
            print(f'{name} {date}: OK', flush=True)
    cal = api.query('trade_cal', exchange='SSE', start_date='20211201', end_date='20211231')
    if '20211231' not in cal.loc[cal.is_open.astype(int) == 1, 'cal_date'].values:
        raise ValueError('Tushare trade_cal is missing an expected trading day')
    print('trade_cal 202112: OK', flush=True)


def get_calendar(api):
    calendars = []
    for year in range(int(START[:4]), int(ANCHOR[:4]) + 1):
        cal = api.query('trade_cal', exchange='SSE', start_date=max(START, f'{year}0101'),
                        end_date=min(ANCHOR, f'{year}1231'))
        calendars.extend(cal.loc[cal.is_open.astype(int) == 1, 'cal_date'])
    return pd.DatetimeIndex(pd.to_datetime(sorted(set(calendars)), format='%Y%m%d'))


def get_monthly_members(api, calendar, work):
    months = pd.period_range('2008-05', pd.Timestamp(calendar[-1]).to_period('M'), freq='M')
    snapshots = []
    for month in months:
        cache = work / f'csi300-{month}.csv'
        if cache.is_file():
            frame = pd.read_csv(cache, dtype=str)
        else:
            frame = api.query('index_weight', index_code='399300.SZ',
                              start_date=month.start_time.strftime('%Y%m%d'),
                              end_date=min(month.end_time.strftime('%Y%m%d'), ANCHOR))
            if not frame.empty:
                temporary = cache.with_suffix('.tmp')
                frame.to_csv(temporary, index=False)
                temporary.replace(cache)
        if frame.empty:
            if month == months[-1] and month.end_time > calendar[-1]:
                continue
            raise ValueError(f'Missing Tushare CSI 300 membership for {month}')
        last = frame.trade_date.max()
        codes = frame.loc[frame.trade_date == last, 'con_code'].unique()
        if len(codes) != 300:
            alternate = api.query('index_weight', index_code='000300.SH',
                                  start_date=month.start_time.strftime('%Y%m%d'),
                                  end_date=min(month.end_time.strftime('%Y%m%d'), ANCHOR))
            same_date = alternate.loc[alternate.trade_date == last, 'con_code'].unique()
            if len(same_date) == 300:
                codes = same_date
            elif not 290 <= len(codes) <= 300:
                raise ValueError(f'Only {len(codes)} members in the CSI 300 snapshot for {last}')
            if len(codes) < 300:
                print(f'Incomplete CSI 300 snapshot {last}: {len(codes)}/300 members', flush=True)
        snapshots.append((pd.to_datetime(last, format='%Y%m%d'), codes))
        if month.month == 12:
            print(f'Checked monthly CSI 300 snapshots through {month}', flush=True)

    # Only carry forward snapshots that were already available before the session.
    intervals = []
    for position, (as_of, codes) in enumerate(snapshots):
        first = calendar.searchsorted(as_of, side='right')
        next_as_of = snapshots[position + 1][0] if position + 1 < len(snapshots) else calendar[-1]
        end = calendar.searchsorted(next_as_of, side='right') - 1
        if first <= end:
            intervals.extend((code[-2:] + code[:6], calendar[first], calendar[end]) for code in codes)
    return pd.DataFrame(intervals, columns=['code', 'start', 'end'])


def cache_stock(api, name, ts_code, raw):
    cache = raw / f'{name}-{ts_code}.csv'
    if cache.is_file():
        return pd.read_csv(cache, dtype={'trade_date': str})
    frame = api.query(name, ts_code=ts_code, start_date=START, end_date=ANCHOR)
    if frame.duplicated('trade_date').any():
        raise ValueError(f'Duplicated {name} history for {ts_code}')
    temporary = cache.with_suffix('.tmp')
    frame.to_csv(temporary, index=False)
    temporary.replace(cache)
    return frame


def adjust_bars(bars, adjustments):
    factors = adjustments.sort_values('trade_date').set_index('trade_date').adj_factor
    merged = bars.merge(factors, on='trade_date', how='left', validate='one_to_one')
    if merged.adj_factor.isna().any() or (factors <= 0).any():
        raise ValueError('A bar has missing or invalid adjustment factors')
    scale = merged.adj_factor / factors.iloc[-1]
    merged[PRICES] = merged[PRICES].mul(scale, axis=0)
    merged['volume'] = merged.vol.mul(100).div(scale)
    merged['vwap'] = merged.amount.mul(1000).div(merged.volume.where(merged.volume > 0))
    merged['date'] = pd.to_datetime(merged.trade_date, format='%Y%m%d')
    return merged[['date', *PRICES, 'volume', 'vwap']].sort_values('date')


def build_snapshot(api, members_file, output):
    if output.exists():
        raise FileExistsError(output)
    work = output.with_name(output.name + '.work')
    raw, csv_dir, provider = work / 'raw', work / 'csv', work / 'qlib'
    for directory in (raw, csv_dir):
        directory.mkdir(parents=True, exist_ok=True)
    calendar = get_calendar(api)
    if members_file is None:
        members = get_monthly_members(api, calendar, raw)
        membership_source = 'Tushare index_weight, monthly snapshot effective on the following session'
    else:
        members = pd.read_csv(members_file, sep='\t', header=None, names=['code', 'start', 'end'],
                              parse_dates=['start', 'end'])
        membership_source = 'Qlib-format dated membership file'
    if not members.code.str.fullmatch(r'(SH|SZ)\d{6}', case=False).all():
        raise ValueError('Expected Qlib CSI 300 instrument codes')
    members.code = members.code.str.upper()
    evaluation_days = calendar[(calendar >= '2009-01-01') & (calendar <= '2022-02-28')]
    counts = pd.Series({day: members.loc[(members.start <= day) & (day <= members.end), 'code'].nunique()
                        for day in evaluation_days})
    if counts.empty or counts.min() < 280:
        raise ValueError('CSI 300 membership is incomplete during the required period')
    members = members.loc[(members.start <= calendar[-1]) & (members.end >= calendar[0])]

    codes = sorted(members.code.unique())
    def export(code):
        ts_code = f'{code[2:]}.{code[:2]}'
        bars = cache_stock(api, 'daily', ts_code, raw)
        source = ts_code
        if bars.empty and ts_code in OLD_CODES:
            new_code, changed = OLD_CODES[ts_code]
            renamed = cache_stock(api, 'daily', new_code, raw)
            if not renamed.empty and (renamed.trade_date < changed).any():
                bars, source = renamed, new_code
        if bars.empty:
            return code, None, 'no daily history'
        adjustments = cache_stock(api, 'adj_factor', source, raw)
        if adjustments.empty:
            return code, None, 'no adjustment factors'
        adjusted = adjust_bars(bars, adjustments)
        if not adjusted.date.isin(calendar).all():
            raise ValueError(f'Tushare bars outside the SSE calendar for {ts_code}')
        adjusted.to_csv(csv_dir / f'{code}.csv', index=False)
        return code, source, None

    print(f'Preparing {len(codes)} instruments across {len(calendar)} sessions', flush=True)
    unavailable, aliases = {}, {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        for offset in range(0, len(codes), 20):
            batch = codes[offset:offset + 20]
            for code, source, missing in executor.map(export, batch):
                if missing:
                    unavailable[code] = missing
                    print(f'{code}: {missing}; retain membership and leave bars missing', flush=True)
                elif source != f'{code[2:]}.{code[:2]}':
                    aliases[code] = source
            print(f'Prepared {offset + len(batch)}/{len(codes)} instruments', flush=True)
    if len(unavailable) > 10:
        raise ValueError(f'Tushare has no usable history for {len(unavailable)} constituent codes')

    DumpDataAll(csv_path=str(csv_dir), qlib_dir=str(provider), max_workers=2,
                exclude_fields='date').dump()
    calendar_file = provider / 'calendars' / 'day.txt'
    dumped_calendar = pd.DatetimeIndex(pd.to_datetime(calendar_file.read_text().splitlines()))
    if not dumped_calendar.equals(calendar):
        raise ValueError('Qlib export calendar does not match Tushare trade_cal')
    membership_path = provider / 'instruments' / 'csi300.txt'
    members.to_csv(membership_path, sep='\t', index=False, header=False, date_format='%Y-%m-%d')
    shutil.copyfile(calendar_file, calendar_file.with_name('day_future.txt'))
    manifest = {'source': 'Tushare daily, adj_factor, trade_cal', 'start': START, 'anchor': ANCHOR,
                'created_utc': datetime.now(timezone.utc).isoformat(), 'instruments': len(codes),
                'instruments_with_bars': len(codes) - len(unavailable),
                'unavailable': unavailable, 'historical_aliases': aliases,
                'sessions': len(calendar), 'constituents_per_day': [int(counts.min()), int(counts.max())],
                'membership_source': membership_source,
                'csi300_sha256': hashlib.sha256(membership_path.read_bytes()).hexdigest(),
                'raw_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted(raw.glob('*.csv'))}}
    (provider / 'snapshot.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    provider.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', action='store_true')
    membership = parser.add_mutually_exclusive_group()
    membership.add_argument('--members', type=Path)
    membership.add_argument('--monthly-members', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--requests-per-minute', type=int, default=45)
    args = parser.parse_args()
    if args.requests_per_minute < 1:
        parser.error('--requests-per-minute must be positive')
    if not args.probe and (args.output is None or not (args.members or args.monthly_members)):
        parser.error('--output and a membership source are required when building a snapshot')
    token = os.environ.get('TUSHARE_TOKEN', '').strip()
    if not token:
        parser.error('set TUSHARE_TOKEN in the local environment')
    api = Tushare(token, args.requests_per_minute)
    if args.probe:
        probe(api)
    else:
        build_snapshot(api, args.members, args.output)


if __name__ == '__main__':
    main()
