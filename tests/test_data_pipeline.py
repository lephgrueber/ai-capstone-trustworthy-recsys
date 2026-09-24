"""Small end-to-end fixtures exercise temporal semantics and the real eval contract."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import pyarrow.parquet as pq

from eval.harness import load_evaluation_examples, evaluate, SavedPredictionsRecommender
from trustworthy_recsys.data.audit import audit_ratings
from trustworthy_recsys.data.pipeline import run_pipeline
from trustworthy_recsys.data.split import choose_cutoffs
from trustworthy_recsys.data.reporting import code_provenance


def stamp(day):
    return int(pd.Timestamp(f"2020-01-{day:02d}", tz="UTC").timestamp())


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root/"raw"
        self.raw.mkdir()
        self.records = [
            (1,1,5,1), (1,2,2,2), (1,3,4,3), (1,4,5,5),
            (2,1,4,1), (2,3,2,2), (2,4,4,4), (2,5,2,6),
            (3,1,5,3), (3,3,5,5), (4,1,2,1), (4,3,5,5),
        ]
        self.write_ratings(self.records)
        (self.raw/"movies.csv").write_text("movieId,title,genres\n" + "".join(f"{i},Movie {i} (2000),Drama\n" for i in range(1,7)), encoding="utf-8")
        (self.raw/"links.csv").write_text("movieId,imdbId,tmdbId\n" + "".join(f"{i},00{i},{i if i<6 else ''}\n" for i in range(1,7)), encoding="utf-8")
        (self.raw/"tags.csv").write_text(f"userId,movieId,tag,timestamp\n1,1,  thoughtful  ,{stamp(1)}\n1,2,   ,{stamp(2)}\n1,3,later,{stamp(3)}\n", encoding="utf-8")
        self.manifest = self.root/"checksums.txt"
        self.refresh_manifest()
        self.readme = self.root/"README.txt"
        self.readme.write_text("Synthetic MovieLens-shaped test fixture; no real users.\n", encoding="utf-8")

    def write_ratings(self, records):
        (self.raw/"ratings.csv").write_text("userId,movieId,rating,timestamp\n" + "".join(f"{u},{i},{r},{stamp(d)}\n" for u,i,r,d in records), encoding="utf-8")

    def refresh_manifest(self):
        self.manifest.write_text("".join(f"{hashlib.md5(p.read_bytes()).hexdigest()}  {p.name}\n" for p in sorted(self.raw.glob('*.csv'))), encoding="utf-8")

    def run_fixture(self, name="run", chunk_size=2):
        output = self.root/name
        run_pipeline(self.raw, output, self.manifest, self.readme,
                     explicit=("2020-01-03", "2020-01-05"), chunk_size=chunk_size, dataset_name="test-fixture")
        return output

    def test_boundaries_labels_cold_start_and_harness_compatibility(self):
        before = {p.name: p.read_bytes() for p in self.raw.glob('*.csv')}
        output = self.run_fixture()
        folder = output/"calendar"
        train, val, test = [pq.read_table(folder/f"{part}.parquet").to_pandas() for part in ("train", "validation", "test")]
        self.assertEqual((len(train),len(val),len(test)), (5,3,4))
        self.assertLess(train.timestamp_utc.max(), pd.Timestamp("2020-01-03", tz="UTC"))
        self.assertEqual(val.timestamp_utc.min(), pd.Timestamp("2020-01-03", tz="UTC"))
        self.assertEqual(test.timestamp_utc.min(), pd.Timestamp("2020-01-05", tz="UTC"))
        examples = load_evaluation_examples(folder/"test_all.jsonl")
        by_user = {e.user_id: e for e in examples}
        self.assertEqual(set(by_user), {"1","3","4"})
        self.assertEqual(by_user['1'].history_items, ['1'])
        self.assertEqual(by_user['3'].history_items, [])
        self.assertEqual(by_user['4'].history_items, [])
        self.assertEqual(by_user['1'].relevant_items, ['4'])
        warm = load_evaluation_examples(folder/"test_warm.jsonl")
        self.assertEqual([(e.user_id,e.relevant_items) for e in warm], [('4',['3'])])
        validation = load_evaluation_examples(folder/"validation_warm.jsonl")
        self.assertEqual([(e.user_id,e.relevant_items) for e in validation], [('1',['3'])])
        # A fixed fixture prediction checks real harness integration, not model quality.
        scores, _ = evaluate(examples, SavedPredictionsRecommender({'1':['4'], '3':[], '4':['3']}), 1, 1)
        self.assertAlmostEqual(scores['recall@1'], 2/3)
        stats = json.loads((output/"statistics.json").read_text())
        self.assertEqual(stats['splits']['calendar']['evaluation']['test']['excluded_no_positive_labels'], 1)
        self.assertEqual(stats['tags']['blank_tags_removed'],1)
        tags = pq.read_table(folder/"train_tags.parquet").to_pandas()
        self.assertEqual(tags.tag.tolist(), ['thoughtful'])
        self.assertEqual(pq.read_table(output/'links.parquet').to_pandas().imdb_id.iloc[0], '001')
        manifest = json.loads((output/'manifest.json').read_text())
        self.assertIn('trustworthy_recsys/data/pipeline.py', manifest['code']['package_source_sha256'])
        self.assertIn('eval/harness.py', manifest['code']['package_source_sha256'])
        for relative, details in manifest['outputs'].items():
            self.assertEqual(hashlib.sha256((output/relative).read_bytes()).hexdigest(), details['sha256'])
        self.assertFalse((output/'INCOMPLETE').exists())
        self.assertEqual(before, {p.name:p.read_bytes() for p in self.raw.glob('*.csv')})

    def test_reproducible_records_across_batch_sizes_and_no_overwrite(self):
        a, b = self.run_fixture('a',1), self.run_fixture('b',7)
        self.assertEqual(json.loads((a/'statistics.json').read_text()), json.loads((b/'statistics.json').read_text()))
        for name in ['train.parquet','validation.parquet','test.parquet']:
            self.assertTrue(pq.read_table(a/'calendar'/name).equals(pq.read_table(b/'calendar'/name)))
        for name in ['validation_all.jsonl','validation_warm.jsonl','test_all.jsonl','test_warm.jsonl']:
            self.assertEqual((a/'calendar'/name).read_bytes(),(b/'calendar'/name).read_bytes())
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.run_fixture('a')

    def test_duplicate_at_batch_boundary_and_bad_hash_fail(self):
        self.write_ratings([self.records[0],self.records[0],*self.records[1:]])
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.run_fixture()
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, 'pair'):
            audit_ratings(self.raw/'ratings.csv',set(range(1,7)),1)
        self.assertFalse((self.root/'run').exists())

    def test_quantile_cutoffs_keep_whole_days_and_shared_test_period(self):
        day = stamp(1)//86400
        days = {day:70,day+1:10,day+2:10,day+3:10}
        a = choose_cutoffs(days,(.8,.1,.1))
        b = choose_cutoffs(days,(.7,.2,.1))
        self.assertEqual(a,(stamp(3),stamp(4)))
        self.assertEqual(b,(stamp(2),stamp(4)))
        with self.assertRaises(ValueError):choose_cutoffs(days,(.7,.2,.2))
        with self.assertRaises(ValueError):choose_cutoffs(days,explicit=('2020-01-05','2020-01-03'))


if __name__ == '__main__':
    unittest.main()
