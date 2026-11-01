import unittest

from run_eval import parse_top_node_buffers


class ParseTopNodeBuffersTest(unittest.TestCase):
    def test_shared_and_temp_counts_from_postgres_line(self):
        counts = parse_top_node_buffers("Buffers: shared hit=17 read=65377, temp written=15289")

        self.assertEqual(counts, {
            "shared_hit_blocks": 17,
            "shared_read_blocks": 65377,
            "shared_dirtied_blocks": 0,
            "shared_written_blocks": 0,
            "temp_read_blocks": 0,
            "temp_written_blocks": 15289,
        })

    def test_parses_all_counters_from_top_node_and_ignores_child_line(self):
        output = """Hash Join  (cost=1.0..2.0 rows=1 width=8)
  Buffers: shared hit=17 read=65377 dirtied=3 written=4, temp read=5 written=15289
  -> Seq Scan on child  (cost=0.0..1.0 rows=1 width=4)
        Buffers: shared hit=999 read=999
Planning:
  Buffers: shared hit=77
Execution Time: 10.0 ms"""

        counts = parse_top_node_buffers(output)

        self.assertEqual(counts, {
            "shared_hit_blocks": 17,
            "shared_read_blocks": 65377,
            "shared_dirtied_blocks": 3,
            "shared_written_blocks": 4,
            "temp_read_blocks": 5,
            "temp_written_blocks": 15289,
        })


if __name__ == "__main__":
    unittest.main()
