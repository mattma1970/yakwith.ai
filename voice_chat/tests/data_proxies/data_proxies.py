import unittest
from voice_chat.utils import DataProxy


class TestDataProxy(unittest.TestCase):
    def test_get_all(self):
        self.assertEqual(
            len(DataProxy.get_all("data_proxy_tests", "./")),
            2,
            f"Expected 2 got {len(DataProxy.get_all('data_proxy_tests','./'))}",
        )

    def test_client_authorized(self):
        self.assertTrue(
            DataProxy.verify_client_authorized("user_1_key", "data_proxy_tests", "./"),
            "Failed client_authorized",
        )

    def test_client_not_authorized(self):
        self.assertFalse(
            DataProxy.verify_client_authorized(
                "user_1_key_2", "data_proxy_tests", "./"
            ),
            "Returned true for user that was not authorized.",
        )

    def test_service_configured(self):
        self.assertEqual(
            DataProxy.get_3p_service_configs(
                "user_1_key", "data_proxy_tests", "assemblyai", "data_proxy_tests", "./"
            )["api_token"],
            "assembly_token",
            "Failed service_config_test",
        )
